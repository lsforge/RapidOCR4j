package io.github.lsforge;

import io.github.lsforge.entity.OcrConfig;
import io.github.lsforge.entity.OcrResult;
import io.github.lsforge.entity.Pair;
import io.github.lsforge.entity.TupleResult;
import io.github.lsforge.utils.OpencvLoader;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * RapidOCR 纯算法单元测试：不依赖 ONNX 模型，仅验证排序、过滤、结果封装逻辑。
 */
public class RapidOCRUnitTest {

    private static RapidOCR engine;

    @BeforeAll
    static void init() {
        OpencvLoader.loadOpencvLib();
        // 构造时仍会初始化 ONNX 会话，但可通过关闭 det/cls/rec 避免实际推理。
        OcrConfig config = new OcrConfig();
        config.Global.setUseDet(false);
        config.Global.setUseCls(false);
        config.Global.setUseRec(false);
        engine = RapidOCR.create(config);
    }

    @Test
    void sortedBoxesShouldGroupByLineAndSortByX() {
        List<Point[]> boxes = new ArrayList<>();
        boxes.add(box(100, 50));  // 第一行右侧
        boxes.add(box(10, 200));  // 第二行右侧
        boxes.add(box(10, 50));   // 第一行左侧
        boxes.add(box(10, 100));  // 第二行左侧

        List<Point[]> sorted = engine.sortedBoxes(boxes);

        Assertions.assertEquals(4, sorted.size());
        assertBox(sorted.get(0), 10, 50);
        assertBox(sorted.get(1), 100, 50);
        assertBox(sorted.get(2), 10, 100);
        assertBox(sorted.get(3), 10, 200);
    }

    @Test
    void sortedBoxesShouldTolerateSmallYJitterWithinSameLine() {
        List<Point[]> boxes = new ArrayList<>();
        boxes.add(box(200, 55));
        boxes.add(box(100, 50));
        boxes.add(box(300, 58));

        List<Point[]> sorted = engine.sortedBoxes(boxes);

        Assertions.assertEquals(3, sorted.size());
        assertBox(sorted.get(0), 100, 50);
        assertBox(sorted.get(1), 200, 55);
        assertBox(sorted.get(2), 300, 58);
    }

    @Test
    void sortedBoxesShouldReturnEmptyForEmptyInput() {
        Assertions.assertTrue(engine.sortedBoxes(new ArrayList<>()).isEmpty());
    }

    @Test
    void filterResultShouldDropLowConfidenceAndBlankText() {
        Point[] box = box(0, 0);
        List<Point[]> boxes = Arrays.asList(box, box, box, box);
        List<TupleResult> recRes = Arrays.asList(
                new TupleResult("hello", 0.9f, null),
                new TupleResult("   ", 0.9f, null),
                new TupleResult("world", 0.3f, null),
                new TupleResult("ok", 0.6f, null)
        );

        Pair<List<Point[]>, List<TupleResult>> filtered = engine.filterResult(boxes, recRes);

        Assertions.assertEquals(2, filtered.getLeft().size());
        Assertions.assertEquals("hello", filtered.getRight().get(0).getText());
        Assertions.assertEquals("ok", filtered.getRight().get(1).getText());
    }

    @Test
    void getFinalResShouldHandleNullBoxesAndRecRes() {
        OcrResult result = engine.getFinalRes(null, null, null, 0.1, 0, 0, 0);
        Assertions.assertNotNull(result);
        Assertions.assertTrue(result.getRecRes().isEmpty());
    }

    @Test
    void getFinalResShouldWrapOnlyClsRes() {
        List<Pair<String, Float>> clsRes = Collections.singletonList(Pair.of("180", 0.95f));
        OcrResult result = engine.getFinalRes(null, clsRes, null, 0.1, 0, 0.01, 0);

        Assertions.assertEquals(1, result.getRecRes().size());
        Assertions.assertEquals("180", result.getRecRes().get(0).getText());
    }

    private static Point[] box(double x, double y) {
        return new Point[]{
                new Point(x, y),
                new Point(x + 10, y),
                new Point(x + 10, y + 5),
                new Point(x, y + 5)
        };
    }

    private static void assertBox(Point[] box, double x, double y) {
        Assertions.assertEquals(x, box[0].x, 0.001);
        Assertions.assertEquals(y, box[0].y, 0.001);
    }
}
