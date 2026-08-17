package io.github.lsforge.ch_ppocr_cls;

import io.github.lsforge.entity.Pair;

import java.util.ArrayList;
import java.util.List;

public class ClsPostProcess {
    private String[] labelList;

    public ClsPostProcess(String[] labelList) {
        this.labelList = labelList;
    }

    // 处理预测结果（行主序扁平，形状 [N, numClasses]），返回标签和概率的配对列表
    public List<Pair<String, Float>> call(float[] preds, int numClasses) {
        List<Pair<String, Float>> decodeOut = new ArrayList<>();
        int num = preds.length / numClasses;
        for (int i = 0; i < num; i++) {
            int base = i * numClasses;
            int idx = this.argMax(preds, base, numClasses);
            decodeOut.add(Pair.of(labelList[idx], preds[base + idx]));
        }
        return decodeOut;
    }

    // 辅助方法，找出数组中 [from, from+length) 内最大值的索引
    private int argMax(float[] array, int from, int length) {
        int bestIdx = 0;
        float max = array[from];
        for (int i = 1; i < length; i++) {
            if (array[from + i] > max) {
                max = array[from + i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
