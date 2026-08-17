package io.github.lsforge.entity;

import lombok.Data;

@Data
public class OrtInferConfig {
    public int intraOpNumThreads; // 单线程操作线程数
    public int interOpNumThreads; // 多线程操作线程数
    public boolean useCuda; // 是否使用 CUDA
    public int deviceId; // 显卡编号
    public boolean useDml; // 是否使用 DML
    public String modelPath; // 模型路径
    public boolean useArena;

    /**
     * 从 Global 引擎配置 + 模块模型路径组装推理会话配置（引擎参数全局共享一份）。
     */
    public static OrtInferConfig from(OcrConfig.GlobalConfig global, String modelPath) {
        OrtInferConfig config = new OrtInferConfig();
        config.intraOpNumThreads = global.intraOpNumThreads;
        config.interOpNumThreads = global.interOpNumThreads;
        config.useCuda = global.useCuda;
        config.deviceId = global.deviceId;
        config.useDml = global.useDml;
        config.useArena = global.useArena;
        config.modelPath = modelPath;
        return config;
    }
}
