import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

# 创建保存可视化的目录
os.makedirs("perm_visualizations", exist_ok=True)

def visualize_permutation(perm, title, grid_shape, orig_shape=None, save_path=None):
    """
    可视化 perm 操作的效果 - 每个单元格显示原始位置索引
    
    参数:
    perm: 一维置换数组，perm[i] = j 表示新位置 i 的元素来自原始位置 j
    title: 图像标题
    grid_shape: 可视化网格的形状 (行, 列)
    orig_shape: 原始数据的形状 (用于计算原始行列位置)，若为 None 则使用 grid_shape
    save_path: 保存路径，None 表示不保存
    """
    n = len(perm)
    if orig_shape is None:
        orig_shape = grid_shape
    
    # 创建原始位置的行号和列号
    orig_rows = np.zeros(n, dtype=int)
    orig_cols = np.zeros(n, dtype=int)
    for pos in range(n):
        orig_rows[pos] = pos // orig_shape[1]  # 行位置
        orig_cols[pos] = pos % orig_shape[1]   # 列位置
    
    # 创建新网格
    grid_rows = np.zeros(grid_shape, dtype=int)
    grid_cols = np.zeros(grid_shape, dtype=int)
    
    # 填充新网格：新位置 (i,j) 的值 = 原始位置的行列信息
    flat_idx = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if flat_idx < n:
                orig_pos = perm[flat_idx]
                grid_rows[i, j] = orig_rows[orig_pos]
                grid_cols[i, j] = orig_cols[orig_pos]
            else:
                grid_rows[i, j] = -1  # 填充无效位置
                grid_cols[i, j] = -1
            flat_idx += 1
    
    # 创建自定义颜色映射 (行号用蓝色通道，列号用红色通道)
    def create_custom_cmap(max_row, max_col):
        colors = []
        for r in range(max_row + 1):
            for c in range(max_col + 1):
                # 行号 -> 蓝色强度，列号 -> 红色强度
                red = min(1.0, c / max_col * 1.2)   # 列号主导红色
                blue = min(1.0, r / max_row * 1.2)  # 行号主导蓝色
                green = 0.2  # 固定绿色增加区分度
                colors.append((red, green, blue))
        return LinearSegmentedColormap.from_list("custom", colors, N=len(colors))
    
    max_row = orig_shape[0] - 1
    max_col = orig_shape[1] - 1
    cmap = create_custom_cmap(max_row, max_col)
    
    # 动态计算图像尺寸 (每个单元格至少1.1英寸)
    cell_size = 1.1  # 每个单元格的英寸大小
    figsize = (grid_shape[1] * cell_size, grid_shape[0] * cell_size)
    
    # 动态计算字体大小 (根据网格密度)
    fontsize = cell_size * 72 // 4 #max(1, 28 - (grid_shape[0] + grid_shape[1]) // 2)
    
    # 创建图像
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # 创建RGBA颜色数组 (结合行列信息)
    rgba_grid = np.zeros((*grid_shape, 4))
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            r = grid_rows[i, j]
            c = grid_cols[i, j]
            if r == -1 or c == -1:  # 无效位置
                rgba_grid[i, j] = [0.95, 0.95, 0.95, 1]  # 灰色
            else:
                # 归一化颜色值
                red = min(1.0, c / max_col)
                blue = min(1.0, r / max_row)
                rgba_grid[i, j] = [red, 0.2, blue, 1]
    
    # 显示网格
    im = ax.imshow(rgba_grid, aspect='equal')
    
    # 添加网格线 (更细的线)
    for i in range(1, grid_shape[0]):
        ax.axhline(i - 0.5, color='k', linewidth=0.3, alpha=0.5)
    for j in range(1, grid_shape[1]):
        ax.axvline(j - 0.5, color='k', linewidth=0.3, alpha=0.5)
    
    # 添加原始位置标签 (每个单元格都显示)
    flat_idx = 0
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if flat_idx < n:
                orig_pos = perm[flat_idx]
                # 根据背景色决定文字颜色 (深色背景用白色，浅色用黑色)
                bg_color = rgba_grid[i, j][:3]
                brightness = 0.299*bg_color[0] + 0.587*bg_color[1] + 0.114*bg_color[2]
                text_color = 'white' if brightness < 0.6 else 'black'
                
                ax.text(j, i, str(orig_pos), 
                        ha='center', va='center', 
                        fontsize=fontsize,
                        color=text_color,
                        weight='bold' if fontsize > 6 else 'normal')
            else:
                ax.text(j, i, 'PAD', 
                        ha='center', va='center', 
                        fontsize=fontsize*0.8,
                        color='darkgray')
            flat_idx += 1
    
    # 添加图例说明 (缩小图例)
    legend_elements = [
        patches.Patch(facecolor=[1, 0.2, 0, 1], label='High Column Index (Red)'),
        patches.Patch(facecolor=[0, 0.2, 1, 1], label='High Row Index (Blue)'),
        patches.Patch(facecolor=[0.95, 0.95, 0.95, 1], label='Padding (Gray)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 1.02),
              fontsize=max(8, 14 - grid_shape[0]//4))
    
    # 设置标题和标签 (动态调整字体大小)
    title_fontsize = max(12, 24 - grid_shape[0]//4)
    label_fontsize = max(10, 16 - grid_shape[0]//4)
    
    ax.set_title(f'{title}\nNew Grid Shape: {grid_shape}, Original Shape: {orig_shape}', 
                 fontsize=title_fontsize, pad=20)
    ax.set_xlabel('New Column Index', fontsize=label_fontsize)
    ax.set_ylabel('New Row Index', fontsize=label_fontsize)
    ax.set_xticks(np.arange(grid_shape[1]))
    ax.set_yticks(np.arange(grid_shape[0]))
    
    # 设置刻度标签 (仅当网格较小时显示)
    if grid_shape[1] <= 20:
        ax.set_xticklabels([str(j) for j in range(grid_shape[1])], fontsize=8)
    else:
        ax.set_xticklabels([])
        
    if grid_shape[0] <= 20:
        ax.set_yticklabels([str(i) for i in range(grid_shape[0])], fontsize=8)
    else:
        ax.set_yticklabels([])
    
    # 隐藏刻度线
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    if save_path:
        # 保存时使用更高DPI确保文字清晰
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.close(fig)  # 关闭图像释放内存

# Precompute permutations for Marlin weight and scale shuffling 
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()

# ========== 可视化部分 ==========
print("Generating permutation visualizations with ALL numbers displayed...")

# 1. 可视化主 perm (1024 elements) - 32x32 网格
visualize_permutation(
    perm=_perm.numpy(),
    title="Main Weight Permutation (Marlin Format)",
    grid_shape=(32, 32),
    orig_shape=(32, 32),
    save_path="perm_visualizations/main_perm_FULL.png"
)

# 2. 可视化 scale_perm (64 elements) - 8x8 网格
visualize_permutation(
    perm=np.array(_scale_perm),
    title="Scale Permutation (Grouped by Column)",
    grid_shape=(8, 8),
    orig_shape=(8, 8),
    save_path="perm_visualizations/scale_perm_FULL.png"
)

# 3. 可视化 scale_perm_single (32 elements) - 4x8 网格
visualize_permutation(
    perm=np.array(_scale_perm_single),
    title="Single Scale Permutation (Interleaved Layout)",
    grid_shape=(4, 8),
    orig_shape=(8, 4),  # 原始是8行4列
    save_path="perm_visualizations/scale_perm_single_FULL.png"
)

print("\nAll visualizations generated with EVERY CELL labeled!")
print("Check the 'perm_visualizations/' directory for high-resolution images")