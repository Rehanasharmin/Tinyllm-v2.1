#!/usr/bin/env python3
"""
TinyLLM Production Benchmark Suite
Comprehensive performance and quality testing for TinyLLM v2.1
"""
import torch
import time
import os
import json
from pathlib import Path
from model import TinyLLM

def setup_matplotlib_for_plotting():
    """Setup matplotlib for plotting with proper configuration."""
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt

class TinyLLMBenchmark:
    """Comprehensive benchmark suite for TinyLLM."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_model(self, model_path, version):
        """Load a TinyLLM model."""
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            chars = ckpt['chars']
            max_len = 96 if version == "v2.1" else 192
            
            model = TinyLLM(len(chars), dim=192, n_layers=5, n_heads=6, max_len=max_len)
            model.load_state_dict(ckpt['model'])
            model.eval()
            
            stoi = ckpt.get('stoi', {c: i for i, c in enumerate(chars)})
            itos = ckpt.get('itos', {i: c for c, i in stoi.items()})
            
            self.models[version] = {
                'model': model,
                'chars': chars,
                'stoi': stoi,
                'itos': itos,
                'max_len': max_len
            }
            
            param_count = sum(p.numel() for p in model.parameters())
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            
            print(f"✓ Loaded {version}: {param_count:,} params, {file_size:.1f}MB")
            return True
            
        except FileNotFoundError:
            print(f"⚠ {version} model not found at {model_path}")
            return False
        except Exception as e:
            print(f"✗ Error loading {version}: {e}")
            return False
    
    def generate_response(self, version, prompt, max_new=50, temperature=0.7):
        """Generate response from specified model version."""
        if version not in self.models:
            return None
            
        model_data = self.models[version]
        model, stoi, itos, max_len = (
            model_data['model'], 
            model_data['stoi'], 
            model_data['itos'],
            model_data['max_len']
        )
        
        ids = [stoi.get(c, 0) for c in prompt]
        
        with torch.no_grad():
            for _ in range(max_new):
                x = torch.tensor([ids[-max_len:]])
                logits = model(x)[0, -1] / temperature
                
                # Apply nucleus sampling
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
                
                # Stop at newline or end token
                if itos[nxt] == '\n' or itos[nxt] == '|':
                    break
        
        return ''.join(itos[i] for i in ids)
    
    def performance_benchmark(self, version):
        """Run performance benchmarks."""
        if version not in self.models:
            return None
            
        test_prompts = [
            "Hello!",
            "What is AI?", 
            "Tell me a joke",
            "How are you?",
            "What can you do?"
        ]
        
        # Format prompts based on version
        if version == "v2.1":
            formatted_prompts = [f"User: {p}\nAssistant: " for p in test_prompts]
        else:
            formatted_prompts = [f"<|user|>{p}<|bot|>" for p in test_prompts]
        
        times = []
        throughputs = []
        
        print(f"\n🔄 Performance testing {version}...")
        for i, prompt in enumerate(formatted_prompts):
            start = time.perf_counter()
            output = self.generate_response(version, prompt, max_new=50)
            elapsed = time.perf_counter() - start
            
            if output:
                tokens = len(output) - len(prompt)
                tps = tokens / elapsed if elapsed > 0 else 0
                
                times.append(elapsed * 1000)  # ms
                throughputs.append(tps)
                
                print(f"  Test {i+1}: {elapsed*1000:.1f}ms, {tps:.1f} tok/s")
        
        return {
            'times': times,
            'throughputs': throughputs,
            'avg_time': sum(times) / len(times),
            'avg_throughput': sum(throughputs) / len(throughputs)
        }
    
    def quality_benchmark(self, version):
        """Test response quality and consistency."""
        if version not in self.models:
            return None
            
        quality_tests = [
            ("Who are you?", "identity"),
            ("What color do you like?", "personality"),
            ("What did I say earlier?", "memory"),
            ("What time is it?", "time"),
            ("Hello", "greeting"),
            ("Explain quantum physics", "knowledge"),
            ("Tell me about cats", "general")
        ]
        
        print(f"\n🎯 Quality testing {version}...")
        results = []
        
        for question, category in quality_tests:
            if version == "v2.1":
                prompt = f"User: {question}\nAssistant: "
            else:
                prompt = f"<|user|>{question}<|bot|>"
                
            output = self.generate_response(version, prompt, max_new=50)
            
            if output:
                # Extract response
                if version == "v2.1":
                    response = output.split('Assistant: ')[-1].split('\n')[0].strip()
                else:
                    response = output.split('<|bot|>')[-1].replace('<|end|>', '').strip()
                
                word_count = len(response.split())
                char_count = len(response)
                
                results.append({
                    'question': question,
                    'category': category,
                    'response': response,
                    'word_count': word_count,
                    'char_count': char_count
                })
                
                print(f"  {category}: {word_count} words - '{response[:50]}{'...' if len(response) > 50 else ''}'")
        
        return results
    
    def create_comparison_report(self):
        """Create comprehensive comparison report."""
        plt = setup_matplotlib_for_plotting()
        
        # Create comparison charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TinyLLM v2.0 vs v2.1 - Production Benchmark', fontsize=18, fontweight='bold')
        
        versions = list(self.results.keys())
        
        # Performance comparison
        if len(versions) >= 2:
            perf_data = {v: self.results[v]['performance'] for v in versions if 'performance' in self.results[v]}
            
            if perf_data:
                version_names = list(perf_data.keys())
                avg_times = [perf_data[v]['avg_time'] for v in version_names]
                avg_throughput = [perf_data[v]['avg_throughput'] for v in version_names]
                
                # Latency comparison
                ax1.bar(version_names, avg_times, color=['#ff7f7f', '#4CAF50'])
                ax1.set_ylabel('Average Latency (ms)')
                ax1.set_title('Response Latency Comparison')
                for i, v in enumerate(avg_times):
                    ax1.text(i, v + max(avg_times)*0.02, f'{v:.1f}ms', ha='center', fontweight='bold')
                
                # Throughput comparison
                ax2.bar(version_names, avg_throughput, color=['#ff7f7f', '#4CAF50'])
                ax2.set_ylabel('Tokens per Second')
                ax2.set_title('Generation Throughput')
                for i, v in enumerate(avg_throughput):
                    ax2.text(i, v + max(avg_throughput)*0.02, f'{v:.1f}', ha='center', fontweight='bold')
        
        # Quality metrics
        if 'v2.1' in self.results and 'quality' in self.results['v2.1']:
            quality_data = self.results['v2.1']['quality']
            word_counts = [item['word_count'] for item in quality_data]
            categories = [item['category'] for item in quality_data]
            
            # Word count distribution
            ax3.hist(word_counts, bins=range(1, max(word_counts)+2), alpha=0.7, color='#4CAF50', edgecolor='black')
            ax3.axvline(x=12, color='red', linestyle='--', label='Target (8-15 words)')
            ax3.axvline(x=8, color='red', linestyle='--')
            ax3.set_xlabel('Word Count')
            ax3.set_ylabel('Frequency')
            ax3.set_title('v2.1 Response Length Distribution')
            ax3.legend()
            
            # Category performance
            category_counts = {}
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            ax4.bar(category_counts.keys(), category_counts.values(), color='#45B7D1')
            ax4.set_ylabel('Test Count')
            ax4.set_title('Test Categories Coverage')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('/workspace/tiny_llm/benchmark_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Benchmark comparison chart saved")
        plt.close()
    
    def save_results(self, filename='benchmark_results.json'):
        """Save benchmark results to JSON."""
        output_path = Path('/workspace/tiny_llm') / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ Results saved to {filename}")
    
    def run_full_benchmark(self):
        """Run comprehensive benchmarks on all available models."""
        print("=" * 60)
        print("🚀 TinyLLM Production Benchmark Suite")
        print("=" * 60)
        
        # Try to load both models
        v21_loaded = self.load_model('tiny_llm_v2_1.pt', 'v2.1')
        v20_loaded = self.load_model('tiny_llm.pt', 'v2.0')
        
        if not v21_loaded and not v20_loaded:
            print("❌ No models found to benchmark")
            return
        
        # Run benchmarks
        for version in self.models.keys():
            print(f"\n🔍 Benchmarking {version}...")
            
            # Performance tests
            perf_results = self.performance_benchmark(version)
            if perf_results:
                self.results[version] = {'performance': perf_results}
            
            # Quality tests
            quality_results = self.quality_benchmark(version)
            if quality_results:
                if version not in self.results:
                    self.results[version] = {}
                self.results[version]['quality'] = quality_results
        
        # Generate reports
        print("\n📊 Generating reports...")
        self.create_comparison_report()
        self.save_results()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 60)
        
        for version in self.results.keys():
            result = self.results[version]
            print(f"\n{version.upper()}:")
            
            if 'performance' in result:
                perf = result['performance']
                print(f"  ⚡ Avg Latency: {perf['avg_time']:.1f}ms")
                print(f"  🚀 Avg Throughput: {perf['avg_throughput']:.1f} tok/s")
            
            if 'quality' in result:
                quality = result['quality']
                word_counts = [q['word_count'] for q in quality]
                avg_words = sum(word_counts) / len(word_counts)
                print(f"  📝 Avg Response Length: {avg_words:.1f} words")
                print(f"  📊 Total Test Cases: {len(quality)}")
        
        if 'v2.1' in self.results and 'v2.0' in self.results:
            # Compare improvements
            v21_perf = self.results['v2.1'].get('performance', {})
            v20_perf = self.results['v2.0'].get('performance', {})
            
            if v21_perf and v20_perf:
                time_improvement = ((v20_perf['avg_time'] - v21_perf['avg_time']) / v20_perf['avg_time']) * 100
                throughput_improvement = ((v21_perf['avg_throughput'] - v20_perf['avg_throughput']) / v20_perf['avg_throughput']) * 100
                
                print(f"\n🎯 v2.1 IMPROVEMENTS:")
                if time_improvement > 0:
                    print(f"  ⏱️ Latency: {time_improvement:.1f}% faster")
                if throughput_improvement > 0:
                    print(f"  📈 Throughput: {throughput_improvement:.1f}% better")
        
        print("\n" + "=" * 60)


def main():
    """Run the benchmark suite."""
    benchmark = TinyLLMBenchmark()
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()