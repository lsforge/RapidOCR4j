package io.github.lsforge;

import io.github.lsforge.entity.OcrConfig;
import io.github.lsforge.utils.OpencvLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 性能基准（非单元测试，surefire 不会拾取）。运行方式见 README 性能实测章节。
 *
 * 单线程不同 intraOpNumThreads 与多线程并发调用同一实例的组合测试，
 * 输出吞吐、平均延迟，以及同期的进程内存（Windows RSS，含 ONNXRuntime/OpenCV
 * native 内存）与 Java 堆占用峰值；末尾验证 close() 的内存释放效果。
 * 默认使用 src/test/resources/test.png（2481x3508 大图）。
 */
public class Benchmark {

    static final String SRC_IMG = "src/test/resources/test.png";

    // 内存采样（后台监控线程写入，runCase 前清零、后汇总）
    static final List<Long> rssKb = new ArrayList<>();
    static final List<Long> heapKb = new ArrayList<>();
    static volatile boolean monitoring = false;

    public static void main(String[] args) throws Exception {
        System.out.println("JVM: " + System.getProperty("java.version")
                + ", 逻辑核: " + Runtime.getRuntime().availableProcessors());
        long baselineRss = sampleRssKb();
        System.out.printf("基线 RSS（未加载库/模型）: %d MB%n", baselineRss / 1024);

        OpencvLoader.loadOpencvLib();
        byte[] img = loadImage();
        // 简单预热类加载 + OpenCV + 模型会话
        try (RapidOCR warm = RapidOCR.create()) {
            warm.run(img);
        }
        startMonitor();

        // {并发数, intraOpNumThreads(-1=ORT默认), 每线程迭代}
        int[][] cases = {
                {1, -1, 5}, {1, 4, 5}, {1, 8, 5},
                {2, -1, 5}, {4, -1, 5},
                {2, 8, 5}, {4, 4, 5}, {8, 2, 5},
        };
        System.out.println("| 并发 | intra | 总次数 | 总耗时(s) | 吞吐(img/s) | 平均延迟(ms) | RSS峰值(MB) | 堆峰值(MB) |");
        System.out.println("|---|---|---|---|---|---|---|---|");
        for (int[] c : cases) {
            runCase(img, c[0], c[1], c[2]);
        }

        // close() 释放验证：创建实例→推理→关闭，观察 RSS 回落
        System.gc();
        Thread.sleep(2000);
        long beforeClose = latestRss();
        try (RapidOCR ocr = RapidOCR.create()) {
            ocr.run(img);
        }
        System.gc();
        Thread.sleep(2000);
        long afterClose = latestRss();
        System.out.printf("%nclose() 验证: 实例关闭后 RSS %d MB → %d MB（native 分配器缓存部分内存属正常）%n",
                beforeClose / 1024, afterClose / 1024);

        monitoring = false;
    }

    static void runCase(byte[] img, int concurrency, int intra, int iters) throws Exception {
        OcrConfig config = new OcrConfig();
        if (intra > 0) {
            config.Global.setIntraOpNumThreads(intra);
        }
        try (RapidOCR ocr = RapidOCR.create(config)) {
            // 会话预热（不计入）
            ocr.run(img);
            ocr.run(img);

            rssKb.clear();
            heapKb.clear();
            long t0 = System.nanoTime();
            AtomicInteger errors = new AtomicInteger();
            Thread[] threads = new Thread[concurrency];
            for (int t = 0; t < concurrency; t++) {
                threads[t] = new Thread(() -> {
                    for (int i = 0; i < iters; i++) {
                        try {
                            ocr.run(img);
                        } catch (Exception e) {
                            errors.incrementAndGet();
                            e.printStackTrace();
                        }
                    }
                });
                threads[t].start();
            }
            for (Thread th : threads) {
                th.join();
            }
            double sec = (System.nanoTime() - t0) / 1e9;
            int total = concurrency * iters - errors.get();
            System.out.printf("| %d | %s | %d | %.1f | %.2f | %.0f | %d | %d |%n",
                    concurrency, intra < 0 ? "默认" : String.valueOf(intra),
                    total, sec, total / sec, sec * 1000 / total,
                    maxOf(rssKb) / 1024, maxOf(heapKb) / 1024);
        }
    }

    /** 每 2 秒采样一次进程 RSS（tasklist，含 native）与 Java 堆使用量 */
    static void startMonitor() {
        monitoring = true;
        Thread t = new Thread(() -> {
            while (monitoring) {
                try {
                    synchronized (rssKb) {
                        rssKb.add(sampleRssKb());
                        Runtime rt = Runtime.getRuntime();
                        heapKb.add((rt.totalMemory() - rt.freeMemory()) / 1024);
                    }
                } catch (Exception ignored) {
                    // 采样失败跳过本轮
                }
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }, "mem-monitor");
        t.setDaemon(true);
        t.start();
    }

    static long maxOf(List<Long> list) {
        synchronized (rssKb) {
            long max = 0;
            for (long v : list) {
                if (v > max) {
                    max = v;
                }
            }
            return max;
        }
    }

    static long latestRss() throws Exception {
        return sampleRssKb();
    }

    /** Windows 下用 tasklist 查询本进程工作集（KB） */
    static long sampleRssKb() throws Exception {
        String pid = ManagementFactory.getRuntimeMXBean().getName().split("@")[0];
        Process p = Runtime.getRuntime().exec(
                new String[]{"tasklist", "/FI", "PID eq " + pid, "/FO", "CSV", "/NH"});
        String line;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream(), "GBK"))) {
            line = br.readLine();
        }
        if (line == null || !line.contains(pid)) {
            throw new IllegalStateException("tasklist 输出异常: " + line);
        }
        // CSV 形如 "java.exe","12345","Console","1","1,234,567 K"
        String[] parts = line.split("\",\"");
        String mem = parts[parts.length - 1].replace("\"", "")
                .replace(" K", "").replace(",", "").trim();
        return Long.parseLong(mem);
    }

    static byte[] loadImage() throws Exception {
        if (!Files.exists(Paths.get(SRC_IMG))) {
            throw new IllegalStateException("测试图不存在: " + SRC_IMG);
        }
        Mat src = Imgcodecs.imread(SRC_IMG, Imgcodecs.IMREAD_COLOR);
        MatOfByte buf = new MatOfByte();
        Imgcodecs.imencode(".jpg", src, buf);
        byte[] data = new byte[(int) buf.total() * buf.channels()];
        buf.get(0, 0, data);
        src.release();
        buf.release();
        return data;
    }
}
