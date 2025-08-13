import numpy as np
import torch

def get_perm_tile4(tile=4):
    """
    基于 Marlin 的构造思路，构造 tile=4 的 perm（示意版）。
    返回：
      perm (torch.LongTensor) : 一维索引数组
      interleave (np.array)   : 用于演示的 interleave 向量
    """
    perm = []
    copies = 4                # 原代码里 j in range(4)，保留为 4 份复制（示意用）
    for i in range(2 * tile):  # 原来是 range(32) (2*16)，这里用 2*tile
        perm1 = []
        col = i // 4          # 保留原来的 i//4 划分方式（可观察到 col 在 0..tile/2-1 之间）
        t = i % 4
        # 模仿原逻辑生成行索引（为避免越界，用 % tile）
        # 原代码构造了 4 个 row（通过 t 与 t+4），这里用 tile 版本并取模
        rows = [
            (2 * t) % tile,
            (2 * t + 1) % tile,
            (2 * (t + tile // 4)) % tile,
            (2 * (t + tile // 4) + 1) % tile
        ]
        # block = 0/1 表示选取列的两半区（类似 +8*block 的思路，这里用 tile//2）
        for block in [0, 1]:
            for row in rows:
                idx = tile * row + col + (tile // 2) * block
                perm1.append(idx)
        # 把 perm1 复制到 4 个相邻的 block（模拟原来的 +256*j）
        base_block_size = tile * tile  # 16 for tile=4
        for j in range(copies):
            perm.extend([p + base_block_size * j for p in perm1])

    perm = np.array(perm, dtype=np.int64)
    # 原实现对每个 8 元组做 interleave (偶位先，奇位后)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)
    # 按每 8 列为一组做 interleave 重排（示意）
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    return torch.from_numpy(perm), interleave

def demo_tile4():
    tile = 16
    perm, interleave = get_perm_tile4(tile)

    print("perm length:", perm.numel())
    print("perm (flat):", perm.numpy()[:256])

    # 为了演示，我们构造一个 'res' 行，它的列数等于 perm 长度
    # 用简单的数字填充，便于观察索引如何被重排
    res = np.arange(perm.numel(), dtype=np.int32).reshape(1, -1)
    print("\n=== 原始 res (第一行，按列显示) ===")
    print(res)

    # 为了对照，我们也把 res 按每 8 列切分并显示（和 perm reshape(-1,8) 对应）
    print("\n=== res.reshape(-1,8)（重排前，每行 8 个元素的块） ===")
    print(res.reshape(4, 16, 16)[0])

    # 应用 perm —— 模拟代码中 res.reshape((-1, _perm.numel()))[:, _perm]
    reshaped = res.reshape((-1, perm.numel()))
    perm_np = perm.numpy()
    after = reshaped[:, perm_np].reshape(res.shape)

    print("\n=== 应用 perm 之后的 res（第一行） ===")
    print(after)

    # 为了更清楚看出每个 8 元组被 interleave（偶位先奇位后）的效果，展示 perm reshape 结果
    print("\n=== perm reshape 到每 8 个一组（显示每组内的索引顺序） ===")
    print(perm_np.reshape(4, 16, 16)[0])

if __name__ == "__main__":
    demo_tile4()
