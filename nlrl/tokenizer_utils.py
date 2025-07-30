"""
Utility functions for handling tokenizer loading, especially for Qwen models
"""

from transformers import AutoTokenizer
import json
import os


def load_tokenizer_safe(model_path):
    """
    Safely load tokenizer with fallbacks for different model types.
    
    Args:
        model_path: Path to the model
        
    Returns:
        Loaded tokenizer
    """
    is_qwen = "qwen" in model_path.lower()
    
    # Try different loading strategies
    strategies = []
    
    if is_qwen:
        # For Qwen models, try these strategies in order
        strategies = [
            # Strategy 1: use_fast=False with trust_remote_code
            {"use_fast": False, "trust_remote_code": True},
            # Strategy 2: Standard loading with trust_remote_code
            {"trust_remote_code": True},
            # Strategy 3: use_fast=False without trust_remote_code
            {"use_fast": False},
            # Strategy 4: Force specific tokenizer class
            {"use_fast": False, "trust_remote_code": True, "force_download": True},
        ]
    else:
        # For non-Qwen models, use standard loading
        strategies = [
            {},  # Default loading
            {"use_fast": False},  # Fallback to slow tokenizer
        ]
    
    last_error = None
    for i, kwargs in enumerate(strategies):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            
            # Verify tokenizer works
            test_text = "test"
            _ = tokenizer.encode(test_text)
            
            if is_qwen and i > 0:
                print(f"Loaded Qwen tokenizer using fallback strategy {i+1}")
            
            return tokenizer
            
        except Exception as e:
            last_error = e
            continue
    
    # If all strategies failed, try one more thing for Qwen
    if is_qwen:
        try:
            # Check if we need to manually specify the tokenizer class
            # This is a last resort for problematic Qwen models
            from transformers import PreTrainedTokenizerFast
            
            # Try to load tokenizer config to see what's expected
            tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, 'r') as f:
                    config = json.load(f)
                    
                # Try to use a generic fast tokenizer
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                print("Loaded Qwen tokenizer using PreTrainedTokenizerFast directly")
                return tokenizer
        except Exception as e:
            last_error = e
    
    # If everything failed, raise the last error
    raise RuntimeError(f"Failed to load tokenizer from {model_path}. Last error: {last_error}")


def get_chat_template_function(model_path):
    """
    Get the appropriate chat template function for the model.
    
    Args:
        model_path: Path to the model
        
    Returns:
        Chat template function
    """
    if "qwen" in model_path.lower():
        from nlrl.llm_call import qwen3_instruct_format
        return qwen3_instruct_format
    else:
        from nlrl.llm_call import llama3_instruct_format
        return llama3_instruct_format