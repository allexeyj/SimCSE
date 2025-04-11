import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
# Remove unused BERT imports if desired
# from transformers import RobertaTokenizer # Keep if needed elsewhere
# from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead

# Import ModernBERT components
from transformers import ModernBertModel, ModernBertPreTrainedModel, ModernBertConfig
# Import the correct head for ModernBERT Masked LM
from transformers.models.modern_bert.modeling_modern_bert import ModernBertDecoderHead

from transformers.activations import gelu # Keep if used, though ModernBert might use its own activations internally
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput

# Keep MLPLayer, Similarity, Pooler, cl_init, cl_forward, sentemb_forward as they are.
# They are designed to work with the output structure (hidden states) and config,
# which ModernBertModel provides in a compatible way for these functions.

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        # Ensure config has hidden_size attribute (ModernBertConfig does)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation. Note: ModernBERT might not have a dedicated pooler layer output.
           We rely on taking the first token's hidden state.
    'cls_before_pooler': Equivalent to 'cls' for models without a dedicated pooler layer like ModernBERT.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        # Update assertion comment if needed, but types remain valid conceptual pooling strategies
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # ModernBertModel output doesn't have 'pooler_output' attribute in BaseModelOutput
        # pooler_output = getattr(outputs, 'pooler_output', None) # Check just in case, but likely None
        hidden_states = outputs.hidden_states # Make sure output_hidden_states=True is passed if needed

        # For ModernBERT, 'cls' and 'cls_before_pooler' extract the same thing:
        # the hidden state of the first token (usually CLS).
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            # Handle cases where CLS token might not be at index 0 if tokenizer changes?
            # Assuming standard CLS token at index 0 for now.
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            # Masked average pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9) # Prevent division by zero
            return sum_embeddings / sum_mask
        elif self.pooler_type == "avg_first_last":
             if hidden_states is None or len(hidden_states) < 2:
                 raise ValueError("Pooling type 'avg_first_last' requires output_hidden_states=True and at least 2 hidden layers.")
            # hidden_states[0] is embeddings, hidden_states[1] is the first layer
             first_hidden = hidden_states[1] # First layer output
             last_hidden = hidden_states[-1] # Last layer output
             input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
             sum_embeddings = torch.sum(((first_hidden + last_hidden) / 2.0) * input_mask_expanded, 1)
             sum_mask = input_mask_expanded.sum(1)
             sum_mask = torch.clamp(sum_mask, min=1e-9)
             return sum_embeddings / sum_mask
        elif self.pooler_type == "avg_top2":
            if hidden_states is None or len(hidden_states) < 3: # Need embedding + 2 layers
                raise ValueError("Pooling type 'avg_top2' requires output_hidden_states=True and at least 2 hidden layers.")
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(((last_hidden + second_last_hidden) / 2.0) * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function. (Should be compatible)
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        # This MLP is added *after* pooling, so it's independent of the base model's pooler
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    # cls.init_weights() # This is called by from_pretrained automatically


def cl_forward(cls,
    encoder, # Now this will be an instance of ModernBertModel
    input_ids=None,
    attention_mask=None,
    token_type_ids=None, # ModernBERT might not use token_type_ids, check config/usage
    position_ids=None,
    head_mask=None, # ModernBERT might not support head_mask
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    # ModernBERT specific args (add if needed, but likely handled by tokenizer/model)
    # sliding_window_mask=None,
    # indices=None,
    # cu_seqlens=None,
    # max_seqlen=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1) # 2: pair instance; 3: pair instance with a hard negative

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)

    # ModernBERT often doesn't use token_type_ids. Check if your tokenizer provides them
    # and if the model expects them. If not, set to None or handle appropriately.
    # Defaulting to passing it if provided.
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Determine if hidden states are needed based *both* on pooler type and explicit request
    compute_hidden_states = True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else \
                            (output_hidden_states if output_hidden_states is not None else False)


    # Get raw embeddings from ModernBERT encoder
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        # token_type_ids=token_type_ids, # Pass only if ModernBERT model uses them
        position_ids=position_ids,
        # head_mask=head_mask, # ModernBERT might not support head_mask
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=compute_hidden_states, # Pass flag determined above
        return_dict=True,
        # Add other ModernBERT specific args if needed
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        # Pass the same relevant arguments as above
        mlm_outputs_raw = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            # MLM calculation only needs last_hidden_state usually
            output_hidden_states=False, # Avoid recomputing unless head needs them
            return_dict=True,
        )
        # We only need the sequence output to pass to the LM head
        mlm_sequence_output = mlm_outputs_raw.last_hidden_state


    # Pooling
    # The Pooler class expects the BaseModelOutputWithPoolingAndCrossAttentions-like object
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls" pooling, add the extra MLP layer
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representations
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Replace local tensors with original ones to preserve gradients
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs * N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    # Calculate similarity
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    # Handle hard negatives in similarity matrix
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    # Create labels for contrastive loss
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Adjust similarity scores for hard negatives if needed
    if num_sent == 3:
        z3_weight = cls.model_args.hard_negative_weight
        # Ensure weights are calculated correctly based on gathered tensors shape
        current_batch_size = z1.size(0) # This is the total batch size after gathering
        weights = torch.tensor(
            [[0.0] * current_batch_size + [0.0] * i + [z3_weight] + [0.0] * (current_batch_size - i - 1) for i in range(current_batch_size)]
            # Original code assumed z1_z3_cos had same dim as z1/z2 batch dim, which is true after gather
            #[[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        # Make sure shapes match for addition
        cos_sim = cos_sim + weights

    # Calculate contrastive loss
    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    # Use the ModernBertDecoderHead (self.lm_head)
    if mlm_input_ids is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        # Pass the sequence output from the MLM run to the LM head
        prediction_scores = cls.lm_head(mlm_sequence_output) # ModernBertDecoderHead takes hidden states
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    # Prepare output
    if not return_dict:
        # Mimic original output tuple structure
        output = (cos_sim,) + (outputs.hidden_states, outputs.attentions) if compute_hidden_states else (None, None)
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states, # Will be None if not computed
        attentions=outputs.attentions,     # Will be None if not computed
    )


def sentemb_forward(
    cls,
    encoder, # Now this will be an instance of ModernBertModel
    input_ids=None,
    attention_mask=None,
    token_type_ids=None, # ModernBERT might not use token_type_ids
    position_ids=None,
    head_mask=None, # ModernBERT might not support head_mask
    inputs_embeds=None,
    labels=None, # Not used in sentence embedding mode
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    # ModernBERT specific args (add if needed)
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # Determine if hidden states are needed
    compute_hidden_states = True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else \
                            (output_hidden_states if output_hidden_states is not None else False)


    # Pass arguments to ModernBERT encoder
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        # token_type_ids=token_type_ids, # Pass only if needed
        position_ids=position_ids,
        # head_mask=head_mask, # Pass only if needed/supported
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=compute_hidden_states,
        return_dict=True,
         # Add other ModernBERT specific args if needed
    )

    # Apply pooling
    pooler_output = cls.pooler(attention_mask, outputs)

    # Apply MLP if pooler is 'cls' and mlp_only_train is False (evaluation mode)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    # Prepare output
    if not return_dict:
        # Return last_hidden_state, pooler_output, and potentially hidden_states/attentions
        return (outputs.last_hidden_state, pooler_output) + (outputs.hidden_states, outputs.attentions)

    # Return a BaseModelOutput-like structure
    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states, # Will be None if not computed
        attentions=outputs.attentions,     # Will be None if not computed
    )


# Define the main class using ModernBERT components
class ModernBertForCL(ModernBertPreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"position_ids"] # Keep if applicable to ModernBERT
    # ModernBERT uses RoPE, position_ids might be handled differently or optional. Check documentation.
    _keys_to_ignore_on_load_missing = [r"position_ids"] # Tentatively keep
    _supports_sdpa = True # Add if ModernBERT supports SDPA (check transformers version/docs)


    def __init__(self, config: ModernBertConfig, *model_args, **model_kargs):
        super().__init__(config)
        self.config = config # Store config explicitly
        # model_kargs likely contains 'model_args' from the user script
        self.model_args = model_kargs["model_args"]

        # Instantiate ModernBertModel
        self.modernbert = ModernBertModel(config)

        # Instantiate the correct LM head if MLM is enabled
        if self.model_args.do_mlm:
            # ModernBert uses ModernBertDecoderHead for MLM
            self.lm_head = ModernBertDecoderHead(config)
        else:
             self.lm_head = None # Explicitly set to None if not used

        # Initialize contrastive learning components (pooler, MLP, similarity)
        cl_init(self, config)

        # Initialize weights, handle tying if necessary (though from_pretrained handles loading)
        # self.post_init() # Call this if needed for weight init/tying logic in PreTrainedModel

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None, # Pass along, cl_forward/sentemb_forward will decide to use it
        position_ids=None,
        head_mask=None,      # Pass along
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,      # Flag to switch between CL and sentence embedding modes
        mlm_input_ids=None,  # MLM specific inputs
        mlm_labels=None,
        # Add other ModernBERT specific args if they need to be passed from the top level
        # sliding_window_mask=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if sent_emb:
            # Call the sentence embedding forward function
            return sentemb_forward(self, self.modernbert, # Pass self.modernbert
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            # Call the contrastive learning forward function
            # Need to explicitly pass output_hidden_states decision based on pooler type
            # cl_forward handles this internally now
            return cl_forward(self, self.modernbert, # Pass self.modernbert
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels, # These are contrastive labels (arange) created inside cl_forward
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, # Let cl_forward decide if needed
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels, # Actual MLM labels
            )
