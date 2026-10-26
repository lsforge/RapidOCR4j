package io.github.lsforge.entity;

import lombok.*;

// OCR 主配置类
@Data
public class OcrConfig {
    public GlobalConfig Global = new GlobalConfig(); // 全局配置
    public DetConfig Det = new DetConfig(); // 检测模块配置
    public ClsConfig Cls = new ClsConfig(); // 分类模块配置
    public RecConfig Rec = new RecConfig(); // 识别模块配置


    // 全局配置类
    @Data
    public static class GlobalConfig {
        public float textScore = 0.5f; // 文本评分阈值
        public boolean useDet = true; // 是否使用检测模块
        public boolean useCls = true; // 是否使用分类模块
        public boolean useRec = true; // 是否使用识别模块
        public int minHeight = 30; // 最小高度
        public float widthHeightRatio = 8; // 宽高比
        public int maxSideLen = 2000; // 最大边长
        public int minSideLen = 30; // 最小边长
        public boolean usePreprocessImg = true; // 是否开启整图 min/max side len 预处理（对齐 Python use_preprocess_img）
        public boolean useVerticalPadding = true; // 细长图是否做垂直 letterbox 填充（对齐 Python use_vertical_padding）

        // 推理引擎参数（全局一份，det/cls/rec 三个模块共享；对齐 Python 端 EngineConfig 全局共享语义）
        public int intraOpNumThreads = -1; // 单线程操作线程数
        public int interOpNumThreads = -1; // 多线程操作线程数
        public boolean useCuda = false; // 是否使用 CUDA
        public int deviceId = 0; // 显卡编号
        public boolean useDml = false; // 是否使用 DML
        public boolean useArena = false; // arena内存池的扩展策略（速度有提升，但内存会剧增，且持续占用，不释放，默认关闭）

        public String opencvLibPath; // opencv环境依赖dll或so目录
    }

    // 检测模块配置类
    @Data
    public static class DetConfig {
        public String modelPath = "models/PP-OCRv6_det_small.onnx"; // 模型路径（默认 PP-OCRv6 det small，对齐 Python 端 rapidocr>=3.9.0）
        public int limitSideLen = 736; // 限制边长
        public String limitType = "min"; // 限制类型
        public float thresh = 0.3f; // 检测阈值
        public float boxThresh = 0.5f; // 边框阈值
        public int maxCandidates = 1000; // 最大候选框数
        public float unclipRatio = 1.6f; // 非极大值抑制后的扩展比例
        public boolean useDilation = true; // 是否使用膨胀操作
        public String scoreMode = "fast"; // 评分模式
    }

    // 分类模块配置类
    @Data
    public static class ClsConfig {
        public String modelPath = "models/ch_ppocr_mobile_v2.0_cls_infer.onnx"; // 模型路径（PP-OCRv6 套件无独立 cls 模型，沿用 v2.0）
        public int[] clsImageShape = {3, 48, 192}; // 分类输入图像形状
        public int clsBatchNum = 6; // 分类批量处理数（对齐 Python 端默认值）
        public float clsThresh = 0.9f; // 分类阈值
        public String[] labelList = {"0", "180"}; // 分类标签列表
    }

    // 识别模块配置类
    @Data
    public static class RecConfig {
        public String modelPath = "models/PP-OCRv6_rec_small.onnx"; // 模型路径（默认 PP-OCRv6 rec small，对齐 Python 端 rapidocr>=3.9.0）
        public int[] recImgShape = {3, 48, 320}; // 识别输入图像形状
        public int recBatchNum = 6; // 识别批量处理数（对齐 Python 端默认值）
        public String recKeysPath; // 字典路径，如果不设置，默认从模型获取
    }
}
