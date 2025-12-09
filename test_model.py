#!/usr/bin/env python3
"""
Quick test script for TinyLLM v2.1 production model
Non-interactive test to verify model functionality
"""
import torch
from model import TinyLLM
import sys

def test_model():
    """Test the v2.1 model with key quality scenarios."""
    print("🔍 Testing TinyLLM v2.1 Production Model")
    print("=" * 50)
    
    try:
        # Load v2.1 model
        ckpt = torch.load('tiny_llm_v2_1.pt', map_location='cpu', weights_only=False)
        chars = ckpt['chars']
        stoi = ckpt.get('stoi', {c: i for i, c in enumerate(chars)})
        itos = ckpt.get('itos', {i: c for c, i in stoi.items()})
        
        model = TinyLLM(len(chars), dim=192, n_layers=5, n_heads=6, max_len=96)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded: {param_count:,} parameters")
        
    except FileNotFoundError:
        print("❌ v2.1 model not found")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test generation function
    def generate_test_response(prompt, max_new=50):
        """Generate test response."""
        ids = [stoi.get(c, 0) for c in prompt]
        
        with torch.no_grad():
            for _ in range(max_new):
                x = torch.tensor([ids[-96:]])  # v2.1 context length
                logits = model(x)[0, -1] / 0.7
                
                # Simple sampling
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumsum < 0.9
                nucleus[1:] = nucleus[:-1].clone()
                nucleus[0] = True
                
                filtered_indices = sorted_indices[nucleus]
                filtered_probs = sorted_probs[nucleus]
                
                if len(filtered_indices) > 0:
                    next_idx = torch.multinomial(filtered_probs, 1)
                    nxt = filtered_indices[next_idx].item()
                else:
                    nxt = sorted_indices[0].item()
                
                ids.append(nxt)
                
                # Stop at newline
                if itos[nxt] == '\n':
                    break
        
        return ''.join(itos[i] for i in ids)
    
    # Test cases for v2.1 quality
    test_cases = [
        ("User: Who are you?\nAssistant: ", "Identity"),
        ("User: What color do you like?\nAssistant: ", "Personality"),
        ("User: What did I say earlier?\nAssistant: ", "Memory"),
        ("User: What time is it?\nAssistant: ", "Time"),
        ("User: Hello!\nAssistant: ", "Greeting")
    ]
    
    print("\n🧪 Running Quality Tests:")
    print("-" * 50)
    
    for i, (prompt, category) in enumerate(test_cases, 1):
        try:
            output = generate_test_response(prompt, max_new=30)
            response = output.split('Assistant: ')[-1].split('\n')[0].strip()
            
            # Check word count (v2.1 target: 8-15 words)
            word_count = len(response.split())
            
            print(f"{i}. {category}:")
            print(f"   Response: '{response}'")
            print(f"   Words: {word_count} {'✓' if 8 <= word_count <= 15 else '⚠'}")
            
        except Exception as e:
            print(f"{i}. {category}: ❌ Error - {e}")
    
    print("\n" + "=" * 50)
    print("✅ Model test completed")
    return True

if __name__ == "__main__":
    test_model()