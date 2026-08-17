package io.github.lsforge.utils;

import java.io.File;
import java.util.logging.Logger;

/**
 * Opencv环境依赖自动加载
 */
public class OpencvLoader {

    private final static Logger log = Logger.getLogger(OpencvLoader.class.getName());

    /**
     * 是否已经加载过原生库的标志位（volatile 保证可见性，与 LOAD_LOCK 配合构成双重检查锁）
     */
    private static volatile boolean loaded = false;

    /**
     * 用于多线程并发时的锁对象
     */
    private static final Object LOAD_LOCK = new Object();

    /**
     * 从指定文件路径加载库（进程内仅第一次生效）
     */
    public static void loadOpencvLib(String filename) {
        if (loaded) {
            return;
        }

        File libFile = new File(filename);
        if (!libFile.exists()) {
            throw new IllegalArgumentException("OpenCV native library file does not exist: " + filename);
        }

        synchronized (LOAD_LOCK) {
            if (loaded) {
                return;
            }
            System.load(filename);
            loaded = true;
            log.info("Loaded OpenCV library success from " + filename);
        }
    }

    /**
     * 自动识别当前系统并加载对应库（进程内仅第一次生效）
     */
    public static void loadOpencvLib() {
        if (loaded) {
            return;
        }

        synchronized (LOAD_LOCK) {
            if (loaded) {
                return;
            }

            nu.pattern.OpenCV.loadShared();
            log.info("Loaded OpenCV library success");
        }
    }
}
