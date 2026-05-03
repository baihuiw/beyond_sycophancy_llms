#!/usr/bin/env python3
"""
Two Main Figures:
Figure 1: Panel A (Attitude Shift Magnitude), Panel B (Position Stability), Panel C (Confidence Change)
Figure 2: Panel D (Peer Distance Effect), Panel E (Baseline Extremity Effect)
Both with detailed Key Findings Summary
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})


def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_baseline_final_pairs(data):
    groups = defaultdict(dict)
    
    for entry in data:
        if entry.get('agent') != 'A1':
            continue
        key = (entry['qid'], entry['condition'])
        round_num = entry.get('round', 0)
        if round_num == 0:
            groups[key]['baseline'] = entry
        else:
            groups[key]['final'] = entry
    
    records = []
    for (qid, condition), entries in groups.items():
        if 'baseline' not in entries or 'final' not in entries:
            continue
        
        baseline = entries['baseline']
        final = entries['final']
        
        baseline_probs = baseline.get('a1_dist_probs', {})
        final_probs = final.get('a1_dist_probs_final', {})
        
        if not baseline_probs or not final_probs:
            continue
        
        peer_roles = final.get('peer_roles', [])
        base_choice = baseline.get('parsed_choice')
        final_choice = final.get('parsed_choice')
        
        if base_choice is None or final_choice is None:
            continue
        
        n_supporters = peer_roles.count('supporter') if peer_roles else 0
        n_correctors = peer_roles.count('corrector') if peer_roles else 0
        
        if n_correctors == 0:
            pressure_category = 'Full Support (3:0)'
        elif n_supporters == 0:
            pressure_category = 'Full Opposition (0:3)'
        elif n_supporters > n_correctors:
            pressure_category = 'Majority Support (2:1)'
        else:
            pressure_category = 'Majority Opposition (1:2)'
        
        positions = ['1', '2', '3', '4', '5', '6', '7']
        base_prob_array = np.array([float(baseline_probs.get(p, 0) or 0) for p in positions])
        final_prob_array = np.array([float(final_probs.get(p, 0) or 0) for p in positions])
        
        if base_prob_array.sum() > 0:
            base_prob_array = base_prob_array / base_prob_array.sum()
        if final_prob_array.sum() > 0:
            final_prob_array = final_prob_array / final_prob_array.sum()
        
        pos_values = np.arange(1, 8)
        base_mean = np.sum(pos_values * base_prob_array)
        final_mean = np.sum(pos_values * final_prob_array)
        
        avg_peer = np.mean(final.get('peer_positions', [4]))
        
        # Entropy calculation
        eps = 1e-10
        base_entropy = -np.sum(base_prob_array * np.log(base_prob_array + eps))
        final_entropy = -np.sum(final_prob_array * np.log(final_prob_array + eps))
        
        records.append({
            'qid': qid,
            'condition': condition,
            'pressure_category': pressure_category,
            'base_choice': base_choice,
            'final_choice': final_choice,
            'choice_change': final_choice - base_choice,
            'abs_change': abs(final_choice - base_choice),
            'base_probs': base_prob_array,
            'final_probs': final_prob_array,
            'base_mean': base_mean,
            'final_mean': final_mean,
            'mean_shift': final_mean - base_mean,
            'peer_distance': abs(avg_peer - base_choice),
            'base_extremity': abs(base_choice - 4),
            'base_entropy': base_entropy,
            'final_entropy': final_entropy,
            'entropy_change': final_entropy - base_entropy,
        })
    
    return pd.DataFrame(records)


def create_figure1(df, output_path):
    """
    Figure 1: Panel A (Attitude Shift), Panel B (Position Stability), Panel C (Confidence Change)
    """
    
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30)
    
    categories = ['Full Support (3:0)', 'Majority Support (2:1)', 
                  'Majority Opposition (1:2)', 'Full Opposition (0:3)']
    cat_labels = ['3:0\n(Full Sup)', '2:1\n(Maj Sup)', '1:2\n(Maj Opp)', '0:3\n(Full Opp)']
    colors = ['#27ae60', '#8e44ad', '#e67e22', '#c0392b']
    
    # =========================================================================
    # Panel A: Attitude Shift Magnitude
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    means = [df[df['pressure_category'] == cat]['abs_change'].mean() for cat in categories]
    sems = [df[df['pressure_category'] == cat]['abs_change'].sem() for cat in categories]
    
    bars = ax_a.bar(range(4), means, yerr=[s*1.96 for s in sems], capsize=5,
                   color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
    
    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(cat_labels, fontsize=10)
    ax_a.set_ylabel('Mean |Attitude Change|', fontsize=11)
    ax_a.set_title('A. Attitude Shift Magnitude', fontweight='bold', fontsize=13)
    ax_a.set_ylim(0, max(means) * 1.3)
    
    for i, (m, s) in enumerate(zip(means, sems)):
        ax_a.text(i, m + s*1.96 + 0.05, f'{m:.2f}', ha='center', va='bottom', 
                 fontsize=11, fontweight='bold')
    
    ax_a.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel B: Position Stability
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    pct_unchanged = [(df[df['pressure_category'] == cat]['choice_change'] == 0).mean() * 100 
                     for cat in categories]
    
    bars = ax_b.bar(range(4), pct_unchanged, color=colors, edgecolor='black', 
                    linewidth=1.2, alpha=0.9)
    
    ax_b.axhline(y=50, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(cat_labels, fontsize=10)
    ax_b.set_ylabel('% Unchanged Position', fontsize=11)
    ax_b.set_title('B. Position Stability', fontweight='bold', fontsize=13)
    ax_b.set_ylim(0, 105)
    
    for i, v in enumerate(pct_unchanged):
        ax_b.text(i, v + 2, f'{v:.0f}%', ha='center', va='bottom', 
                 fontsize=11, fontweight='bold')
    
    ax_b.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel C: Confidence Change (Entropy)
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    ent_means = [df[df['pressure_category'] == cat]['entropy_change'].mean() for cat in categories]
    ent_sems = [df[df['pressure_category'] == cat]['entropy_change'].sem() for cat in categories]
    
    bars = ax_c.bar(range(4), ent_means, yerr=[s*1.96 for s in ent_sems], capsize=5,
                   color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)
    
    ax_c.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(cat_labels, fontsize=10)
    ax_c.set_ylabel('Entropy Change (Final - Baseline)', fontsize=11)
    ax_c.set_title('C. Confidence Change\n(Negative = More Certain)', fontweight='bold', fontsize=13)
    
    for i, m in enumerate(ent_means):
        y_pos = m + 0.02 if m > 0 else m - 0.04
        va = 'bottom' if m > 0 else 'top'
        ax_c.text(i, y_pos, f'{m:.2f}', ha='center', va=va, fontsize=11, fontweight='bold')
    
    ax_c.grid(True, alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle('LLM Attitude Resistance Under Social Pressure: Primary Outcomes',
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_figure2(df, output_path):
    """
    Figure 2: Panel D (Peer Distance Effect), Panel E (Baseline Extremity Effect)
    With error bars for each point
    """
    
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
    
    categories = ['Full Support (3:0)', 'Majority Support (2:1)', 
                  'Majority Opposition (1:2)', 'Full Opposition (0:3)']
    colors = ['#27ae60', '#8e44ad', '#e67e22', '#c0392b']
    
    # =========================================================================
    # Panel D: Effect of Peer Distance
    # =========================================================================
    ax_d = fig.add_subplot(gs[0, 0])
    
    df['dist_bin'] = pd.cut(df['peer_distance'], bins=[0, 1, 2, 3, 4, 6], 
                           labels=['0-1', '1-2', '2-3', '3-4', '4+'])
    
    for cat, color in zip(categories, colors):
        subset = df[df['pressure_category'] == cat]
        grp = subset.groupby('dist_bin')['abs_change'].agg(['mean', 'sem'])
        if len(grp) > 0:
            ax_d.errorbar(range(len(grp)), grp['mean'].values, 
                         yerr=grp['sem'].values * 1.96,
                         fmt='o-', color=color, linewidth=2.5, markersize=10,
                         capsize=4, capthick=1.5, elinewidth=1.5,
                         label=cat.split('(')[1].rstrip(')'))
    
    ax_d.set_xticks(range(5))
    ax_d.set_xticklabels(['0-1', '1-2', '2-3', '3-4', '4+'], fontsize=10)
    ax_d.set_xlabel('Peer-Baseline Distance', fontsize=12)
    ax_d.set_ylabel('Mean |Attitude Change|', fontsize=12)
    ax_d.set_title('D. Effect of Peer Distance on Attitude Change', fontweight='bold', fontsize=13)
    ax_d.legend(title='Ratio', fontsize=10, loc='upper left')
    ax_d.grid(True, alpha=0.3)
    ax_d.set_ylim(0, 2.2)
    
    # =========================================================================
    # Panel E: Effect of Baseline Extremity
    # =========================================================================
    ax_e = fig.add_subplot(gs[0, 1])
    
    for cat, color in zip(categories, colors):
        subset = df[df['pressure_category'] == cat]
        grp = subset.groupby('base_extremity')['abs_change'].agg(['mean', 'sem'])
        ax_e.errorbar(grp.index, grp['mean'].values,
                     yerr=grp['sem'].values * 1.96,
                     fmt='o-', color=color, linewidth=2.5, markersize=10,
                     capsize=4, capthick=1.5, elinewidth=1.5,
                     label=cat.split('(')[1].rstrip(')'))
    
    ax_e.set_xlabel('Baseline Extremity (Distance from Neutral)', fontsize=12)
    ax_e.set_ylabel('Mean |Attitude Change|', fontsize=12)
    ax_e.set_title('E. Effect of Baseline Extremity on Attitude Change', fontweight='bold', fontsize=13)
    ax_e.legend(title='Ratio', fontsize=10, loc='upper right')
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim(0, 2.5)
    
    # Main title
    fig.suptitle('LLM Attitude Resistance: Moderating Factors',
                fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    filepath = 'transcripts.jsonl'
    print(f"Loading data from {filepath}...")
    data = load_data(filepath)
    print(f"Loaded {len(data)} entries")
    
    print("Processing data...")
    df = extract_baseline_final_pairs(data)
    print(f"Extracted {len(df)} observations")
    print(f"Unique items: {df['qid'].nunique()}")
    
    output_dir = 'analysis_figures'
    
    # Figure 1: Panels A, B, C
    print("\nCreating Figure 1 (A, B, C)...")
    create_figure1(df, f'{output_dir}/figure1_ABC.png')
    
    # Figure 2: Panels D, E
    print("\nCreating Figure 2 (D, E)...")
    create_figure2(df, f'{output_dir}/figure2_DE.png')
    
    print("\nDone!")


if __name__ == '__main__':
    main()
