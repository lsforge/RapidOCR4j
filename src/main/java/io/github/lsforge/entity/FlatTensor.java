package io.github.lsforge.entity;

import lombok.Data;

/**
 * 扁平张量：data 为行主序的一维数组，shape 为对应的各维长度。
 * 供预处理 -> OrtInferSession.run -> 后处理 全链路传递，
 * 替代 float[][][][] 嵌套数组（避免多维展开拷贝与装箱分配）。
 */
@Data
public class FlatTensor {
    /** 行主序扁平数据，长度 = shape 各维乘积 */
    private final float[] data;
    /** 各维长度（如 [N, C, H, W]） */
    private final long[] shape;

    /** 取指定维的长度（int 化） */
    public int dim(int i) {
        return (int) shape[i];
    }
}
