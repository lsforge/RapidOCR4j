package io.github.lsforge.utils;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;
import io.github.lsforge.entity.FlatTensor;
import io.github.lsforge.entity.OrtInferConfig;

import java.io.*;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.logging.Logger;

public class OrtInferSession implements AutoCloseable {

    private static final Logger logger = Logger.getLogger(OrtInferSession.class.getName());

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public OrtInferSession(OrtInferConfig ortInferConfig) {
        logger.info("Initializing OrtInferSession...");

        String modelPath = ortInferConfig.getModelPath();
        boolean useCuda = ortInferConfig.useCuda;
        boolean useDirectML = ortInferConfig.useDml;

        // 1、创建 ONNX Runtime 环境
        this.env = OrtEnvironment.getEnvironment("OrtInferSessionEnv");

        SessionOptions sessionOptions = null;
        OrtCUDAProviderOptions providerOptions = null;
        try {
            // 2. 初始化 SessionOptions
            sessionOptions = initSessionOptions(ortInferConfig);
            EnumSet<OrtProvider> availableProviders = OrtEnvironment.getAvailableProviders();
            if (useCuda && availableProviders.contains(OrtProvider.CUDA)) {
                providerOptions = new OrtCUDAProviderOptions(ortInferConfig.getDeviceId());
                // kNextPowerOfTwo（默认值）以 2 的幂数扩展，而 kSameAsRequested 每次扩展的大小与分配请求的大小相同。
                providerOptions.add("arena_extend_strategy", "kNextPowerOfTwo");
                providerOptions.add("cudnn_conv_algo_search", "EXHAUSTIVE");
                providerOptions.add("do_copy_in_default_stream", "1");
                sessionOptions.addCUDA(providerOptions);
                logger.info(String.format("Requested CUDA EP added to session options, deviceId: %d.", ortInferConfig.getDeviceId()));
            }

            if (useDirectML && availableProviders.contains(OrtProvider.DIRECT_ML)) {
                sessionOptions.addDirectML(ortInferConfig.getDeviceId());
                logger.info("Requested DirectML EP - might not be supported in certain Java packages.");
            }

            // 最后添加 CPU
            if (availableProviders.contains(OrtProvider.CPU)) {
                sessionOptions.addCPU(ortInferConfig.useArena);
                logger.info("CPU EP added to session options.");
            }

            // 4. 创建推理会话
            Path path = Paths.get(modelPath);
            if (path.isAbsolute()) {
                if (!path.toFile().exists()) {
                    throw new RuntimeException("模型文件未找到: " + modelPath);
                }
                this.session = env.createSession(modelPath, sessionOptions);
            } else {
                InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream(modelPath);
                if (inputStream == null) {
                    throw new RuntimeException("模型资源未找到: " + modelPath);
                }
                this.session = env.createSession(loadModel(inputStream), sessionOptions);
            }
            inputName = this.getInputNames().get(0);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        } finally {
            closeQuietly(sessionOptions);
            closeQuietly(providerOptions);
        }
        logger.info("OrtInferSession initialization completed.");
    }

    private static void closeQuietly(SessionOptions options) {
        if (options != null) {
            try {
                options.close();
            } catch (Exception ignored) {
                // ignore
            }
        }
    }

    private static void closeQuietly(OrtCUDAProviderOptions options) {
        if (options != null) {
            try {
                options.close();
            } catch (Exception ignored) {
                // ignore
            }
        }
    }

    /**
     * 初始化 SessionOptions
     */
    private SessionOptions initSessionOptions(OrtInferConfig ortInferConfig) throws OrtException {
        SessionOptions sessOpt = new SessionOptions();
        int cpuNums = Runtime.getRuntime().availableProcessors();

        int intraOpNumThreads = ortInferConfig.intraOpNumThreads;
        if (intraOpNumThreads >= 1 && intraOpNumThreads <= cpuNums) {
            sessOpt.setIntraOpNumThreads(intraOpNumThreads);
        }
        int interOpNumThreads = ortInferConfig.interOpNumThreads;
        if (interOpNumThreads >= 1 && interOpNumThreads <= cpuNums) {
            sessOpt.setInterOpNumThreads(interOpNumThreads);
        }
        // 禁用 arena 内存池的扩展策略
        sessOpt.setCPUArenaAllocator(ortInferConfig.useArena);
        // 启用图优化
        sessOpt.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT);
        // 日志等级
        sessOpt.setSessionLogVerbosityLevel(4);
        sessOpt.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL);
        return sessOpt;
    }

    /**
     * 执行推理（扁平数据版，内部热路径使用）
     * 输入直接包装为行主序张量；输出用 getValue() 由 native 批量填充嵌套数组后，
     * 逐平面 System.arraycopy（memcpy 级）拼平成行主序 flat 数组返回。
     * 适用于输出较小的模型（det/cls）。
     * 注：不使用 getFloatBuffer()——其内部为 heap 分配 + 逐元素 put 拷贝，大输出时显著慢于装箱路径。
     *
     * @param data  行主序扁平输入数据
     * @param shape 输入形状（如 [N, C, H, W]）
     * @return 扁平输出张量（data + shape）
     */
    public FlatTensor run(float[] data, long[] shape) throws OrtException {
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)) {
            try (Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                OnnxTensor outTensor = (OnnxTensor) result.get(0);
                TensorInfo info = outTensor.getInfo();
                long[] outShape = info.getShape();
                float[] out = flatten(outTensor.getValue(), outShape);
                return new FlatTensor(out, outShape);
            }
        }
    }

    /**
     * 执行推理，输出保持 getValue() 装箱的嵌套数组形态（不做拼平拷贝）。
     * 适用于输出量大的模型（如 rec 的 [N, T, C]，宽批次可达数十 MB，
     * 拼平的 arraycopy 过路费会显著增加耗时），由调用方直接消费嵌套数组。
     *
     * @param data  行主序扁平输入数据
     * @param shape 输入形状（如 [N, C, H, W]）
     * @return 装箱输出（如 float[][][]），结构与模型输出形状一致
     */
    public Object runBoxed(float[] data, long[] shape) throws OrtException {
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)) {
            try (Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                OnnxTensor outTensor = (OnnxTensor) result.get(0);
                return outTensor.getValue();
            }
        }
    }

    /**
     * 将 getValue() 装箱出的嵌套 float 数组按行主序拼平（逐平面 arraycopy）
     */
    private static float[] flatten(Object boxed, long[] shape) {
        int total = 1;
        for (long d : shape) {
            total *= (int) d;
        }
        float[] out = new float[total];
        int off = 0;
        switch (shape.length) {
            case 1: {  // 一维直接返回
                float[] b = (float[]) boxed;
                return b;
            }
            case 2: {  // 如 cls 输出 [N, C]
                float[][] b = (float[][]) boxed;
                for (float[] row : b) {
                    System.arraycopy(row, 0, out, off, row.length);
                    off += row.length;
                }
                break;
            }
            case 3: {  // 如 rec 输出 [N, T, C]
                float[][][] b = (float[][][]) boxed;
                for (float[][] plane : b) {
                    for (float[] row : plane) {
                        System.arraycopy(row, 0, out, off, row.length);
                        off += row.length;
                    }
                }
                break;
            }
            case 4: {  // 如 det 输出 [N, C, H, W]
                float[][][][] b = (float[][][][]) boxed;
                for (float[][][] ch : b) {
                    for (float[][] plane : ch) {
                        for (float[] row : plane) {
                            System.arraycopy(row, 0, out, off, row.length);
                            off += row.length;
                        }
                    }
                }
                break;
            }
            default:
                throw new IllegalArgumentException("不支持的输出维度: " + shape.length);
        }
        return out;
    }

    /**
     * 获取模型的输入名字列表
     */
    private List<String> getInputNames() throws OrtException {
        List<String> names = new ArrayList<>();
        for (NodeInfo info : session.getInputInfo().values()) {
            names.add(info.getName());
        }
        return names;
    }

    /**
     * 在 ONNX 模型的 metadata 中提取指定 key 对应的数据，split line 后返回
     */
    public List<String> getCharacterList(String key) {
        try {
            Map<String, String> customMetadata = session.getMetadata().getCustomMetadata();
            if (customMetadata.containsKey(key)) {
                String content = customMetadata.get(key);
                // 按行拆分
                return Arrays.asList(content.split("\\r?\\n"));
            }
            return Collections.emptyList();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 判断模型中是否包含指定 key
     */
    public boolean haveKey(String key) {
        try {
            Map<String, String> customMetadata = session.getMetadata().getCustomMetadata();
            return customMetadata.containsKey(key);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException ignored) {
                // ignore
            }
        }
        // OrtEnvironment 为命名单例，多个会话共享同一实例；
        // 关闭它可能导致其他会话异常，因此只关闭 session。
    }

    private byte[] loadModel(InputStream modelInputStream) {
        try (
                InputStream inputStream = modelInputStream;
                ByteArrayOutputStream buffer = new ByteArrayOutputStream()
        ) {
            int nRead;
            byte[] data = new byte[65536];

            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }

            buffer.flush();
            return buffer.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
