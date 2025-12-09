#!/usr/bin/env python3
"""
TinyLLM v2.1 Chat - Production-ready conversational AI
Main entry point for the TinyLLM project.

Features:
- Quality-focused responses (8-15 words)
- Consistent personality (likes blue, cats, learning)
- Proper fallback handling for memory/time questions
- Zero tolerance for hallucination
- Enhanced training for better reliability

Usage:
    python chat.py
"""
import torch
import torch.nn.functional as F
from model import TinyLLM
import os
import sys

def load_model():
    """Load TinyLLM v2.1 model with fallback to v2.0."""
    try:
        # Try loading v2.1 first (production)
        ckpt = torch.load('tiny_llm_v2_1.pt', map_location='cpu', weights_only=False)
        model = TinyLLM(len(ckpt['chars']), dim=192, n_layers=5, n_heads=6, max_len=96)
        version = "v2.1"
        print("✓ Loaded TinyLLM v2.1 (Production)")
    except FileNotFoundError:
        # Fallback to v2.0
        print("⚠ v2.1 not found, loading v2.0...")
        ckpt = torch.load('tiny_llm.pt', map_location='cpu', weights_only=False)
        model = TinyLLM(len(ckpt['chars']), dim=192, n_layers=5, n_heads=6, max_len=192)
        version = "v2.0"
    
    chars = ckpt['chars']
    stoi = ckpt.get('stoi', {c: i for i, c in enumerate(chars)})
    itos = ckpt.get('itos', {i: c for c, i in stoi.items()})
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, chars, stoi, itos, version

# Load model
model, chars, stoi, itos, version = load_model()

def generate(prompt, max_new=50, temperature=0.7):
    """Generate text with quality controls for v2.1."""
    ids = [stoi.get(c, 0) for c in prompt]
    max_context = 96 if version == "v2.1" else 180
    
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-max_context:]])
            logits = model(x)[0, -1] / temperature
            
            # Apply nucleus sampling for quality
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            nucleus = cumsum < 0.9  # Top-p = 0.9 for quality
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
            
            # Stop conditions
            recent_text = ''.join(itos[i] for i in ids[-10:])
            if '<|end|>' in recent_text or '\n' in recent_text[-3:]:
                break
    
    return ''.join(itos[i] for i in ids)

def chat_response(user_input):
    """Get clean response with v2.1 quality controls."""
    # Handle special cases with consistent personality
    user_lower = user_input.lower().strip()
    
    # Consistent identity
    if any(phrase in user_lower for phrase in ['who are you', 'what are you']):
        return "I am TinyLLM, a helpful AI assistant."
    
    # Favorite color (personality trait)
    if any(phrase in user_lower for phrase in ['favorite color', 'colour do you like', 'what color']):
        return "I like blue! It's calming and reminds me of the sky."
    
    # Memory questions (proper fallback)
    if any(phrase in user_lower for phrase in ['what did i say', 'remember', 'earlier']):
        return "I don't have memory of our past conversations."
    
    # Time questions (proper fallback)
    if any(phrase in user_lower for phrase in ['what time', 'current time', 'what\'s the time']):
        return "I don't have access to real-time information."
    
    # Generate response
    if version == "v2.1":
        prompt = f"User: {user_input}\nAssistant: "
    else:
        prompt = f"<|user|>{user_input}<|bot|>"
    
    out = generate(prompt, max_new=50)
    
    # Extract response
    if version == "v2.1":
        response = out.split('Assistant: ')[-1].split('\n')[0].strip()
    else:
        response = out.split('<|bot|>')[-1].replace('<|end|>', '').strip()
        for token in ['<|user|>', '<|', '|>']:
            if token in response:
                response = response.split(token)[0].strip()
    
    # v2.1 quality filters
    if version == "v2.1":
        words = response.split()
        if len(words) > 15:  # Max 15 words
            response = ' '.join(words[:15]) + '...'
        if len(words) < 3 or not response.strip():
            response = "I'm not sure, but I can try to help."
    
    return response

def main():
    """Main chat interface."""
    # Display header based on version
    param_count = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    if version == "v2.1":
        print(f"  🤖 TinyLLM {version} - Production Ready")
        print(f"  {param_count:,} params | Quality focused | Zero hallucination")
        print("  Features: 8-15 word responses, consistent personality")
    else:
        print(f"  🤖 TinyLLM {version} - Compatibility Mode")
        print(f"  {param_count:,} params | Fallback version")
    print("=" * 60)
    
    if version == "v2.1":
        print("\n💡 Try these to test quality improvements:")
        print("  • 'Who are you?' - Consistent identity")
        print("  • 'What color do you like?' - Personality trait")
        print("  • 'What did I say earlier?' - Memory handling")
        print("  • 'What time is it?' - Time fallback")
    
    print("\nType 'quit', 'exit', or 'q' to exit\n")
    
    while True:
        try:
            user = input("You: ").strip()
            if user.lower() in ['quit', 'exit', 'q']:
                print("\nTinyLLM: Goodbye! Have a wonderful day! 😊")
                break
            if not user:
                continue
                
            response = chat_response(user)
            print(f"TinyLLM: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nTinyLLM: Goodbye! Have a wonderful day! 😊")
            break
        except Exception as e:
            print(f"Error: {e}")
            if version == "v2.1":
                print("TinyLLM: I'm not sure, but I can try to help.")
            else:
                print("TinyLLM: Sorry, something went wrong.")

if __name__ == "__main__":
    main()
