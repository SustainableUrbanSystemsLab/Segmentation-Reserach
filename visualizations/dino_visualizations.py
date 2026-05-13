"""
DINO detection visualizations.

Provides enhanced diagnostic visualizations for DINO detection results including:
- Per-prompt breakdown: Separate visualization for each detector
- Filtering stage comparison: Shows detections at each filtering stage
- Detection heatmap: Confidence density map overlaid on image
- Box size distribution: Analysis of detected object sizes
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def save_per_prompt_breakdown(
    img: np.ndarray,
    all_records: list[dict],
    run_id: str,
    config=None,
) -> None:
    """
    Create separate visualizations for each prompt group (detector).
    
    Groups records by prompt_group and creates a subplot grid showing
    detections for each category independently.
    
    Uses lower DPI for efficient rendering.
    """
    dpi = 72  # Use lower DPI to avoid memory issues from matplotlib rendering
    # Group records by prompt
    prompt_groups = {}
    for rec in all_records:
        group = rec.get("prompt_group", "unknown")
        if group not in prompt_groups:
            prompt_groups[group] = []
        prompt_groups[group].append(rec)
    
    if not prompt_groups:
        print("[INFO] No records to visualize per-prompt breakdown")
        return
    
    num_prompts = len(prompt_groups)
    cols = min(3, num_prompts)
    rows = (num_prompts + cols - 1) // cols
    
    h, w = img.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), dpi=dpi)
    if num_prompts == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    prompt_colors = {
        "sidewalk": "#00FF00",
        "sitting": "#FF8800",
        "transit_hub": "#FF55CC",
        "pedestrian_features": "#7CFF3A",
        "sidewalk_surface": "#00E6E6",
        "road_surface": "#A9A9A9",
        "building_roof": "#00FF00",
        "warehouse_roof": "#FF8C00",
        "outdoor_seating": "#00FF00",
        "seated_dining": "#FF8800",
        "standing_gathering": "#00FFFF",
        "furniture": "#FFFF00",
    }
    
    for idx, (prompt_name, records) in enumerate(sorted(prompt_groups.items())):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        ax.imshow(img)
        ax.set_title(f"{prompt_name}\n({len(records)} detections)", fontsize=14, weight="bold")
        ax.axis("off")
        
        color = prompt_colors.get(prompt_name, "#FFFFFF")
        
        # Draw boxes for this prompt
        for rec in records:
            box = rec["box"]
            x1, y1, x2, y2 = box.astype(int)
            score = rec["score"]
            phrase = rec["phrase"]
            
            # Draw box
            rect_w = x2 - x1
            rect_h = y2 - y1
            rect = plt.Rectangle((x1, y1), rect_w, rect_h, 
                                 linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            
            # Draw score label
            ax.text(x1, y1 - 5, f"{score:.2f}", 
                   fontsize=9, color=color, weight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    # Hide unused subplots
    for idx in range(num_prompts, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    
    # Save directly at lower DPI to avoid memory issues
    if config:
        output_path = config.results_dir / "dino_detections" / f"{run_id}_per_prompt_breakdown.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        print(f"[INFO] Saved per-prompt breakdown visualization")
    else:
        print("[ERROR] Config required for per_prompt_breakdown visualization")


def save_filtering_stage_comparison(
    img: np.ndarray,
    unfiltered: list[dict],
    all_records: list[dict],
    filtered_out: list[dict],
    run_id: str,
    config=None,
) -> None:
    """
    Create side-by-side comparison of filtering stages.
    
    Shows:
    - Column 1: All unfiltered detections
    - Column 2: Passed detections (final)
    - Column 3: Top-10 filtered out detections
    
    For very large images, downsamples for matplotlib display to avoid memory errors.
    Uses lower DPI for efficient rendering.
    """
    dpi = 72  # Use lower DPI to avoid memory issues from matplotlib rendering
    h, w = img.shape[:2]
    
    # For very large images, downsample for matplotlib display to avoid memory errors
    display_scale = 1
    img_display = img
    
    if h > 5000 or w > 5000:
        # Downsample to ~2000 pixels max dimension to avoid matplotlib allocation errors
        display_scale = max(h, w) / 2000
        img_display = img[::int(display_scale), ::int(display_scale)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    
    # Helper function to scale box coordinates for display
    def scale_box(box):
        x1, y1, x2, y2 = box.astype(int)
        x1_d = max(0, int(x1 / display_scale))
        y1_d = max(0, int(y1 / display_scale))
        x2_d = min(img_display.shape[1], int(x2 / display_scale))
        y2_d = min(img_display.shape[0], int(y2 / display_scale))
        return x1_d, y1_d, x2_d, y2_d
    
    # Stage 1: All unfiltered detections
    ax1 = axes[0]
    ax1.imshow(img_display)
    ax1.set_title(f"Unfiltered Detections\n({len(unfiltered)} boxes)", 
                 fontsize=12, weight="bold")
    ax1.axis("off")
    
    for rec in unfiltered:
        box = rec["box"]
        x1_d, y1_d, x2_d, y2_d = scale_box(box)
        rect_w = x2_d - x1_d
        rect_h = y2_d - y1_d
        if rect_w > 0 and rect_h > 0:
            rect = plt.Rectangle((x1_d, y1_d), rect_w, rect_h, 
                                 linewidth=1.5, edgecolor="#CCCCCC", facecolor="none", alpha=0.7)
            ax1.add_patch(rect)
    
    # Stage 2: Passed detections (green)
    ax2 = axes[1]
    ax2.imshow(img_display)
    ax2.set_title(f"Passed Detections\n({len(all_records)} boxes)", 
                 fontsize=12, weight="bold", color="green")
    ax2.axis("off")
    
    for rec in all_records:
        box = rec["box"]
        score = rec["score"]
        x1_d, y1_d, x2_d, y2_d = scale_box(box)
        rect_w = x2_d - x1_d
        rect_h = y2_d - y1_d
        if rect_w > 0 and rect_h > 0:
            rect = plt.Rectangle((x1_d, y1_d), rect_w, rect_h, 
                                 linewidth=2, edgecolor="#00FF00", facecolor="none")
            ax2.add_patch(rect)
            
            ax2.text(x1_d, max(0, y1_d - 3), f"{score:.2f}", fontsize=8, color="#00FF00", weight="bold")
    
    # Stage 3: Top-10 filtered out (red)
    ax3 = axes[2]
    ax3.imshow(img_display)
    
    top_filtered = sorted(filtered_out, key=lambda r: float(r.get("score", 0.0)), 
                         reverse=True)[:10] if filtered_out else []
    ax3.set_title(f"Filtered Out (Top-10 shown)\n({len(filtered_out)} total filtered)", 
                 fontsize=12, weight="bold", color="red")
    ax3.axis("off")
    
    for rec in top_filtered:
        box = rec["box"]
        score = rec["score"]
        x1_d, y1_d, x2_d, y2_d = scale_box(box)
        rect_w = x2_d - x1_d
        rect_h = y2_d - y1_d
        if rect_w > 0 and rect_h > 0:
            rect = plt.Rectangle((x1_d, y1_d), rect_w, rect_h, 
                                 linewidth=2, edgecolor="#FF0000", facecolor="none", linestyle="--")
            ax3.add_patch(rect)
            
            ax3.text(x1_d, max(0, y1_d - 3), f"{score:.2f}", fontsize=8, color="#FF0000", weight="bold")
    
    plt.tight_layout()
    
    # Save directly at lower DPI to avoid memory issues
    if config:
        output_path = config.results_dir / "dino_detections" / f"{run_id}_filtering_stages.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        print(f"[INFO] Saved filtering stage comparison visualization")
    else:
        print("[ERROR] Config required for filtering_stage_comparison visualization")


def save_detection_heatmap(
    img: np.ndarray,
    all_records: list[dict],
    run_id: str,
    config=None,
) -> None:
    """
    Create a heatmap showing detection confidence density overlaid on the image.
    
    Uses Gaussian blur to create a smooth confidence density visualization.
    High confidence areas appear bright, low confidence areas are dim.
    For very large images, uses a downsampled version for matplotlib display.
    
    Args:
        config: Configuration object with dino_heatmap_mode setting ("average" or "sum")
    """
    dpi = 72  # Use lower DPI to avoid memory issues from matplotlib rendering
    from scipy import ndimage as ndi
    
    h, w = img.shape[:2]
    
    # Determine heatmap mode from config
    heatmap_mode = "average"
    if config and hasattr(config, "dino_heatmap_mode"):
        heatmap_mode = config.dino_heatmap_mode
    
    # Create confidence heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap_count = np.zeros((h, w), dtype=np.float32) if heatmap_mode == "average" else None
    
    for rec in all_records:
        box = rec["box"]
        score = rec["score"]
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] += score
            if heatmap_mode == "average":
                heatmap_count[y1:y2, x1:x2] += 1
    
    # Apply mode-specific normalization
    if heatmap_mode == "average":
        # Average where multiple boxes overlap
        mask = heatmap_count > 0
        heatmap[mask] = heatmap[mask] / heatmap_count[mask]
        # Normalize with percentile clipping to preserve variation
        # Use 95th percentile to avoid outliers compressing the scale
        p95 = np.percentile(heatmap[mask], 95) if np.any(mask) else 1.0
        heatmap = np.clip(heatmap / max(p95, 0.1), 0, 1)
    else:
        # Sum mode: show detection density (higher = more detections)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
    
    # Apply moderate Gaussian blur to smooth while preserving detail
    heatmap = gaussian_filter(heatmap, sigma=10)
    
    # For very large images, downsample for matplotlib display to avoid memory errors
    display_scale = 1
    img_display = img
    heatmap_display = heatmap
    
    if h > 5000 or w > 5000:
        # Downsample to ~2000 pixels max dimension to avoid matplotlib allocation errors
        display_scale = max(h, w) / 2000
        new_h = max(1, int(h / display_scale))
        new_w = max(1, int(w / display_scale))
        img_display = np.array(np.kron(img[::int(display_scale), ::int(display_scale)], 
                                      np.ones((1, 1, 1), dtype=int)))
        heatmap_display = heatmap[::int(display_scale), ::int(display_scale)]
    
    # Create figure with side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
    
    # Original image with boxes (scaled)
    ax1.imshow(img_display)
    ax1.set_title(f"Original with Boxes\n({len(all_records)} detections)", 
                 fontsize=12, weight="bold")
    ax1.axis("off")
    
    # Draw scaled boxes on downsampled display
    for rec in all_records:
        box = rec["box"]
        x1, y1, x2, y2 = box.astype(int)
        
        # Scale box coordinates for display
        x1_d = max(0, int(x1 / display_scale))
        y1_d = max(0, int(y1 / display_scale))
        x2_d = min(img_display.shape[1], int(x2 / display_scale))
        y2_d = min(img_display.shape[0], int(y2 / display_scale))
        
        rect_w = x2_d - x1_d
        rect_h = y2_d - y1_d
        if rect_w > 0 and rect_h > 0:
            rect = plt.Rectangle((x1_d, y1_d), rect_w, rect_h, 
                                 linewidth=1, edgecolor="#00FF00", facecolor="none")
            ax1.add_patch(rect)
    
    # Heatmap overlay (downsampled to avoid memory issues)
    ax2.imshow(img_display, alpha=0.6)
    im = ax2.imshow(heatmap_display, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    ax2.set_title("Confidence Density Heatmap", fontsize=12, weight="bold")
    ax2.axis("off")
    
    plt.colorbar(im, ax=ax2, label="Confidence Score", fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save directly at lower DPI to avoid memory issues
    if config:
        output_path = config.results_dir / "dino_detections" / f"{run_id}_detection_heatmap.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        print(f"[INFO] Saved detection heatmap visualization")
    else:
        print("[ERROR] Config required for detection_heatmap visualization")



def save_box_size_distribution(
    img: np.ndarray,
    all_records: list[dict],
    run_id: str,
    config=None,
) -> None:
    """
    Create analysis of detected box sizes with distribution charts.
    
    Shows:
    - Pie chart: Distribution of box sizes (small/medium/large)
    - Bar chart: Box sizes by prompt group
    - Scatter plot: Box size vs. confidence score
    
    Uses lower DPI for efficient rendering.
    """
    dpi = 72  # Use lower DPI to avoid memory issues from matplotlib rendering
    if not all_records:
        print("[INFO] No records for box size distribution")
        return
    
    h, w = img.shape[:2]
    img_area = h * w
    
    # Calculate box sizes and categorize
    sizes = []
    sizes_by_group = {}
    scores = []
    group_names = []
    
    for rec in all_records:
        box = rec["box"]
        x1, y1, x2, y2 = box.astype(int)
        box_area = max(1, (x2 - x1) * (y2 - y1))
        box_area_ratio = box_area / img_area
        
        sizes.append(box_area_ratio)
        scores.append(rec["score"])
        group = rec.get("prompt_group", "unknown")
        group_names.append(group)
        
        if group not in sizes_by_group:
            sizes_by_group[group] = []
        sizes_by_group[group].append(box_area_ratio)
    
    sizes = np.array(sizes)
    scores = np.array(scores)
    
    # Categorize as small/medium/large
    small = np.sum(sizes < 0.001)
    medium = np.sum((sizes >= 0.001) & (sizes < 0.01))
    large = np.sum(sizes >= 0.01)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=dpi)
    
    # Pie chart: Size distribution
    ax1 = axes[0, 0]
    size_counts = [small, medium, large]
    size_labels = [f"Small <0.1%\n({small})", 
                   f"Medium 0.1-1%\n({medium})", 
                   f"Large >1%\n({large})"]
    colors_pie = ["#3498db", "#f39c12", "#e74c3c"]
    ax1.pie(size_counts, labels=size_labels, colors=colors_pie, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10, "weight": "bold"})
    ax1.set_title("Box Size Distribution", fontsize=12, weight="bold")
    
    # Bar chart: Average size by group
    ax2 = axes[0, 1]
    group_names_unique = sorted(sizes_by_group.keys())
    avg_sizes = [np.mean(sizes_by_group[g]) * 100 for g in group_names_unique]  # Convert to percentage
    bars = ax2.bar(range(len(group_names_unique)), avg_sizes, color="#2ecc71")
    ax2.set_xticks(range(len(group_names_unique)))
    ax2.set_xticklabels(group_names_unique, rotation=45, ha="right")
    ax2.set_ylabel("Average Area (% of image)", fontsize=10, weight="bold")
    ax2.set_title("Average Box Size by Prompt", fontsize=12, weight="bold")
    ax2.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}%', ha='center', va='bottom', fontsize=9)
    
    # Scatter: Size vs Confidence
    ax3 = axes[1, 0]
    ax3.scatter(sizes * 100, scores, alpha=0.6, s=50, c=scores, cmap="viridis")
    ax3.set_xlabel("Box Area (% of image)", fontsize=10, weight="bold")
    ax3.set_ylabel("Confidence Score", fontsize=10, weight="bold")
    ax3.set_title("Box Size vs. Confidence Score", fontsize=12, weight="bold")
    ax3.grid(True, alpha=0.3)
    
    # Histogram: Size distribution
    ax4 = axes[1, 1]
    ax4.hist(sizes * 100, bins=20, color="#9b59b6", alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Box Area (% of image)", fontsize=10, weight="bold")
    ax4.set_ylabel("Count", fontsize=10, weight="bold")
    ax4.set_title("Box Size Distribution Histogram", fontsize=12, weight="bold")
    ax4.grid(axis="y", alpha=0.3)
    
    # Add statistics text
    stats_text = (
        f"Total detections: {len(all_records)}\n"
        f"Mean size: {np.mean(sizes)*100:.4f}%\n"
        f"Median size: {np.median(sizes)*100:.4f}%\n"
        f"Min size: {np.min(sizes)*100:.6f}%\n"
        f"Max size: {np.max(sizes)*100:.4f}%\n"
        f"Std dev: {np.std(sizes)*100:.4f}%"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    
    # Save directly at lower DPI to avoid memory issues
    if config:
        output_path = config.results_dir / "dino_detections" / f"{run_id}_box_size_distribution.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='png')
        plt.close()
        print(f"[INFO] Saved box size distribution visualization")
    else:
        print("[ERROR] Config required for box_size_distribution visualization")

