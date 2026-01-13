#!/usr/bin/env python3
import sys
sys.path.append('src')

from samplers import MultiContextMixtureSampler

def debug_structure():
    sampler = MultiContextMixtureSampler(
        n_dims=20,
        n_contexts=2,
        n_components=2,
        context_length=8,
        predict_length=1
    )
    
    structure = sampler.get_sequence_structure()
    print("=== SEQUENCE STRUCTURE DEBUG ===")
    print(f"n_contexts: {structure['n_contexts']}")
    print(f"context_length: {structure['context_length']}")
    print(f"predict_length: {structure['predict_length']}")
    print(f"total_length: {structure['total_length']}")
    print(f"predict_position: {structure['predict_position']}")
    print(f"predict_inds: {structure['predict_inds']}")
    print(f"sep_positions: {structure['sep_positions']}")
    
    # 手动计算应该的值
    print("\n=== MANUAL CALCULATION ===")
    total_manual = (2 * (8 + 1)) + (8 - 1) + 1
    print(f"Manual total: {total_manual}")
    
    predict_pos_manual = (2 * (8 + 1)) + (8 - 1)
    print(f"Manual predict_position: {predict_pos_manual}")
    
    predict_inds_manual = list(range(predict_pos_manual + 1, min(predict_pos_manual + 2, total_manual)))
    print(f"Manual predict_inds: {predict_inds_manual}")

if __name__ == "__main__":
    debug_structure()