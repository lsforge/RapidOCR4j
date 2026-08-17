# RapidOCR4j

[![Maven Central](https://img.shields.io/maven-central/v/io.github.lsforge/rapidocr4j.svg)](https://central.sonatype.com/artifact/io.github.lsforge/rapidocr4j)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Java](https://img.shields.io/badge/Java-8%2B-blue.svg)

**纯 Java 实现的多平台 OCR 引擎** —— [RapidOCR](https://github.com/RapidAI/RapidOCR)（PaddleOCR 生态）的 Java 移植版，通过 ONNXRuntime 推理 + OpenCV 图像处理，无需 Python 环境即可获得与 Python 端一致识别效果的离线 OCR 能力。

- **开箱即用**：默认 PP-OCRv6 det/rec small 模型打包在 jar 内，引入依赖即可跑通全流程
- **多语言**：v6 单模型支持约 50 种语言，字典内嵌模型，无需额外字典文件
- **数值对齐**：与 Python 端 rapidocr >= 3.9.0 密集对齐，密集文本图上双端识别结果可做到逐行一致
- **热参数**：阈值等参数支持调用级覆盖，并发调用同一实例时各线程可安全使用不同参数
- **字/词级坐标**：除文本行框外，可返回单字（中文）/单词（英文）级坐标框
- **GPU 加速**：支持 CUDA / DirectML（Windows），零代码切换
- **Java 8+**：兼容老旧运行环境

## 快速开始

**Maven**

```xml
<dependency>
    <groupId>io.github.lsforge</groupId>
    <artifactId>rapidocr4j</artifactId>
    <version>3.9.2</version>
</dependency>
```

**Gradle**

```groovy
implementation 'io.github.lsforge:rapidocr4j:3.9.2'
```

**Hello World**

```java
try (RapidOCR rapidOCR = RapidOCR.create()) {   // AutoCloseable，用完释放推理会话
    OcrResult result = rapidOCR.run("test.jpg");

    System.out.println(result.getStrRes());      // 全部文本（按行拼接）
    for (RecResult rec : result.getRecRes()) {   // 逐行结果
        System.out.println(rec.getText() + " | conf=" + rec.getConfidence());
        // rec.getDtBoxes()：文本行四点坐标
    }
    System.out.printf("total=%.3fs det=%.3fs cls=%.3fs rec=%.3fs%n",
            result.getElapseTime(), result.getDetTime(),
            result.getClsTime(), result.getRecTime());
}
```

## 支持的输入

`run()` 接受五种输入类型，内部统一转换为 BGR Mat 后进入流水线：

```java
run(String imagePath)     // 文件路径（自动兼容中文路径）
run(Path imagePath)
run(byte[] imageData)     // 图片二进制内容（无需落盘）
run(BufferedImage image)  // Java AWT 图像
run(Mat mat)              // OpenCV Mat（内部 clone，不改动调用方原图）
```

流程为 `检测 → 方向分类（0°/180°） → 识别`，任一阶段可通过参数关闭（见下）。

## 调用级热参数（ParamConfig）

阈值与开关类参数支持**单次调用覆盖**，随调用链传递、不写回实例状态：

```java
ParamConfig param = new ParamConfig();
param.setUseCls(false);          // 跳过方向分类（竖排/倒置文本建议保持开启）
param.setTextScore(0.6f);        // 识别置信度过滤阈值
param.setBoxThresh(0.5f);        // 检测框得分阈值
param.setUnclipRatio(1.6f);      // 检测框扩张比率
param.setReturnWordBox(true);    // 返回字/词级坐标
param.setReturnWordLevel(true);  // 英文按单词合并（false 则单字）
OcrResult result = rapidOCR.run("img.png", param);
```

## 配置（OcrConfig）

构造时传入，四组配置与 RapidOCR 的 config.yaml 一一对应：

```java
OcrConfig config = new OcrConfig();
config.Global.setIntraOpNumThreads(4);   // 见下方"性能调优"
RapidOCR rapidOCR = RapidOCR.create(config);
```

**Global（全局）**——推理引擎参数集中在 Global 一份，对 det/cls/rec 三个模型统一生效（对齐 Python 端引擎配置全局共享的语义）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `intraOpNumThreads` / `interOpNumThreads` | -1 | ORT 线程数（-1 交给 ONNX Runtime 默认） |
| `useCuda` / `useDml` / `deviceId` | false / false / 0 | GPU 执行提供者 |
| `useArena` | false | arena 内存池（提速但内存剧增且不释放） |
| `useDet` / `useCls` / `useRec` | true | 三阶段总开关 |
| `textScore` | 0.5 | 识别置信度过滤阈值 |
| `maxSideLen` / `minSideLen` | 2000 / 30 | 整图缩放限制（`usePreprocessImg` 可关闭） |
| `useVerticalPadding` | true | 细长图垂直 letterbox 填充 |
| `minHeight` / `widthHeightRatio` | 30 / 8 | letterbox 触发条件 |
| `opencvLibPath` | null | 非内置平台手动指定 OpenCV 原生库路径 |

**Det / Cls / Rec（模块）**——各模块只保留自身模型参数：`modelPath`（classpath 相对路径或绝对路径）、det 的 `limitSideLen`/`limitType`/`thresh`/`boxThresh`/`unclipRatio` 等、cls 的 `clsThresh`/`clsBatchNum`、rec 的 `recImgShape`/`recBatchNum`。字段含义与 Python 端 config.yaml 完全对应，详见 [config.yaml 参数解释](https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/)。

## 性能调优

- **大图耗时**：检测耗时随输入分辨率近线性增长。默认 `Global.maxSideLen=2000` + `Det.limitType="min"` 意味着大图会以接近原始分辨率进入检测模型。若业务允许牺牲少量精度换速度，可调小 `maxSideLen`（如 1280）或将 `Det.limitType` 改为 `"max"`。

- **并发调用与线程数**：det/cls/rec 三个推理会话的 `intraOpNumThreads` 默认由 ONNX Runtime 决定（通常占用全部 CPU 核）。多线程并发调用同一 `RapidOCR` 实例时，线程总数 = 并发数 × 每会话线程数，会产生线程超订（实测 4 线程并发仅约 1.4× 吞吐）。通过 `Global.intraOpNumThreads` 一处设置全部模块，建议按 `核数 / 并发数` 取值：

```java
OcrConfig config = new OcrConfig();
config.Global.setIntraOpNumThreads(4);  // 例：16 核机器并发 4 路 → 每模块 4 线程
```

  单线程场景无需设置。det 推理的线程扩展性实测参考（16 逻辑核，1984×1312 输入）：4 线程 1265ms / 8 线程 797ms / ORT 默认 747ms——超过物理核数后收益为负。

### 性能实测（并发与内存）

实测环境：Windows 11 / JDK 8 / 16 逻辑核 CPU，输入 `test.png`（2481×3508 大图，检测→分类→识别全流程，默认参数），每组合预热 2 次后计时；内存为测试期间每 2 秒采样一次的进程工作集峰值（RSS，**含 ONNXRuntime/OpenCV 的 native 内存**）与 Java 堆使用峰值：

| 并发数 | intraOpNumThreads | 吞吐 (img/s) | 平均延迟 (ms) | 加速比 | RSS 峰值 (MB) | Java 堆峰值 (MB) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ORT 默认 | 0.48 | 2075 | 1.00× | 1212 | 528 |
| 1 | 4 | 0.37 | 2730 | 0.77× | 1453 | 477 |
| 1 | 8 | 0.48 | 2070 | 1.00× | 1036 | 320 |
| 2 | ORT 默认 | 0.61 | 1635 | 1.27× | 1889 | 618 |
| 4 | ORT 默认 | 0.68 | 1476 | 1.42× | 3937 | 932 |
| 2 | 8 | 0.65 | 1536 | 1.35× | 3338 | 844 |
| 4 | 4 | 0.70 | 1434 | 1.46× | 3815 | 718 |
| 8 | 2 | **0.74** | **1349** | **1.54×** | 5424 | 1742 |

（进程基线 RSS 约 27 MB，未计入上表。）

结论：

- **并发收益有限且递减**：OCR 全流程为 CPU 密集型，并发数翻倍吞吐仅小幅提升（默认线程下 2 路 1.27×、4 路 1.42×），线程超订与 det/cls/rec 三会话争抢是主因。
- **高并发时限流收益显著**：并发 8 路时按 `核数/并发数`（16/8=2）限流 `intraOpNumThreads`，吞吐与平均延迟同时达到最优（0.74 img/s、1349ms）。
- **单线程不要调小线程数**：大图场景 intra=4 反而比默认慢 32%（det 单阶段对线程数敏感），单线程场景保持默认即可。
- **内存随并发近线性增长**：每增加 1 路并发约增加 500~700 MB RSS（整图预处理副本 + det 中间张量 + 推理线程池），8 路并发峰值达 5.4 GB——高并发部署请按并发数预留内存。
- **大头在 native 内存**：Java 堆峰值仅为 RSS 的 1/3 左右（8 路时 1.7 GB vs 5.4 GB），推理数据链路主要在 ONNXRuntime/OpenCV 的 native 堆——**`-Xmx` 限制不住 RSS**，容器化部署应以进程 RSS 而非堆做内存预算。
- **close() 释放验证**：实例创建→推理→close 后进程 RSS 无额外增长（实例级无泄漏迹象），但历史峰值内存不会归还操作系统（native 分配器缓存行为，属正常现象）。
- 复现方式：`src/test/java/io/github/lsforge/Benchmark.java`（非单元测试，surefire 不拾取），`mvn test-compile` 后以 main 方法运行。

## GPU 加速

排除 CPU 版 `onnxruntime` 依赖、改用 `onnxruntime_gpu`（版本对应关系见 [官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)），并置 `useCuda=true`（Windows 亦可用 DirectML：`useDml=true`）：

```xml
<dependency>
    <groupId>io.github.lsforge</groupId>
    <artifactId>rapidocr4j</artifactId>
    <version>3.9.2</version>
    <exclusions>
        <exclusion>
            <groupId>com.microsoft.onnxruntime</groupId>
            <artifactId>onnxruntime</artifactId>
        </exclusion>
    </exclusions>
</dependency>

<!-- 1.18.0 support CUDA 12.x -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.18.0</version>
</dependency>
```

```java
OcrConfig config = new OcrConfig();
config.Global.setUseCuda(true);   // 或 Windows: setUseDml(true)，对三个模型统一生效
RapidOCR rapidOCR = RapidOCR.create(config);
```

## 词级坐标与可视化

开启 `returnWordBox` 后，每行结果的 `wordBoxResult` 携带字/词级内容、坐标与置信度；`returnWordLevel=true` 时英文单字自动合并为单词。对已有结果也可事后合并：

```java
WordBoxResult merged = WordBoxMerger.mergeEnglishWords(rec.getWordBoxResult());
```

可视化绘制（需要字体文件）：

```java
Mat visImg = new VisRes().run("img.png", ocrResult.getRecRes(), "FZYTK.TTF");   // 行级
Mat visWord = new VisRes().runWord("img.png", ocrResult.getRecRes(), "FZYTK.TTF"); // 词级
```


## 与 Python 端对齐

行为对齐基准为 Python 端 [rapidocr](https://github.com/RapidAI/RapidOCR) >= 3.9.0（PP-OCRv6 默认配置），包括：检测框排序与坐标逆变换、整图缩放的 32 对齐算法（银行家舍入）、rec 批次动态宽度、竖排文本条的逆时针转正、空文本行过滤、中英文分词阈值等——密集文本图上双端识别文本可做到逐行一致。

## 许可证

[Apache License 2.0](LICENSE)

**内置模型**：PP-OCRv6 det/rec 与 cls 模型来自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0），打包于 jar 内 `models/` 目录。

**第三方组件**：

| 组件 | 许可证 |
|---|---|
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | MIT |
| [OpenCV](https://github.com/opencv/opencv)（openpnp 打包） | Apache-2.0 |
| Clipper 多边形裁剪（`de.lighti.clipper`，移植版） | Apache-2.0 |
| Lombok | MIT |
