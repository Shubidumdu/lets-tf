<template>
  <div>
    <div>
      <p>머신러닝 모델을 학습시키는 단계는 다음과 같습니다.</p>
      <p>
        <b>작업을 체계적으로 정리:</b>
        <ul>
          <li>회귀 문제인가, 분류 문제인가?</li>
          <li>지도 학습을 수행해야 하는가, 비지도 학습을 수행해야 하는가?</li>
          <li>입력 데이터의 모양이 어떤가? 출력은 어떻게 표시되어야 하는가?</li>
        </ul>
      </p>
      <p>
        <b>데이터 준비:</b>
        <ul>
          <li>가능하면 데이터를 정리하고 수동으로 패턴을 검사합니다.</li>
          <li>학습에 사용하기 전에 데이터를 셔플링합니다.</li>
          <li>신경망에 합당한 범위로 데이터를 정규화합니다. 일반적으로 수치 데이터에는 0~1 또는 -1~1이 적합합니다.</li>
          <li>데이터를 텐서로 변환합니다.</li>
        </ul>
      </p>
      <p>
        <b>모델 빌드 및 실행:</b>
        <ul>
          <li>`tf.sequential`또는 `tf.model`을 사용해 모델을 정의한 후, `tf.layers.*`를 통해 모델에 레이어를 추가합니다.</li>
          <li>옵티마이저, 매개변수(배치 크기, epoch 수)를 선택합니다.</li>
          <li>문제에 적합한 손실 함수(loss func.)를 선택하고 진행 상황을 평가하는 데 도움이 되는 정확도 측정항목을 선택합니다. meanSquaredError는 회귀 문제의 일반적인 loss 함수입니다.</li>
          <li>
            손실이 감수하는지 학습을 모니터링합니다.
          </li>
        </ul>
      </p>
      <p>
        <b>모델 평가</b>
        <ul>
          <li>학습 중에 모니터링할 수 있는 모델의 평가 측정항목을 선택합니다. 학습이 완료되면 예측 품질을 파악할 수 있도록 테스트 예측을 시도합니다.</li>
        </ul>
      </p>
    </div>
    <div ref="plotContainer"></div>
    <div ref="summaryContainer"></div>
    <div ref="trainContainer"></div>
    <div ref="testContainer"></div>
  </div>
</template>
<script lang="ts">
import { defineComponent, onMounted, ref } from "vue";
import {
  layers,
  linspace,
  losses,
  Rank,
  Sequential,
  sequential,
  Tensor,
  tensor2d,
  tidy,
  train,
  util,
} from "@tensorflow/tfjs";
import { render, show } from "@tensorflow/tfjs-vis";
import { getCarsData, Car } from "../api/data";

const createModel = () => {
  // 여기서 우리는 마력 x에 대한 MPG를 예측하고자 합니다.
  // 시퀀셜 모델을 만듭니다.
  // sequential 모델은 입력이 출력으로 곧바로 흘러감을 의미합니다.
  const model = sequential();
  // 단일 Input 레이어를 만듭니다.
  // 최초의 입력은 하나의 값(마력) 뿐입니다.
  model.add(layers.dense({ inputShape: [1], units: 1 }));
  model.add(layers.dense({ units: 50, activation: "sigmoid" }));
  // 하나의 Ouput 레이어를 만듭니다.
  // 마지막에 도출될 출력 역시 하나의 값(MPG) 뿐입니다.
  model.add(layers.dense({ units: 1, activation: "sigmoid" }));
  // sigmoid 함수는 대표적인 "비선형" 활성화 함수입니다.
  return model;
};

const runModel = (ref: HTMLElement) => {
  // 앞서 만든 함수로 모델을 생성합니다.
  const model = createModel();
  // 모델에 대한 요약을 표로 나타냅니다.
  show.modelSummary(ref, model);
  return model;
};

// 아래 함수는, 실제 데이터를 기계가 학습할 수 있도록 '텐서'로 변환해주는 역할을 합니다.
// 이를 수행하는 중에, 데이터 셔플 혹은 노멀라이징을 수행할 수 있습니다.
const convertToTensor = (data: any[]) => {
  return tidy(() => {
    // Step 1. 데이터를 셔플합니다.
    // 학습 중에는 데이터를 여러 작은 하위 집합(=batch)로 분할하기 때문에 이러한 과정이 중요합니다.
    // 셔플을 통해 순서에 의존치 않으며 각 배치에 전체 데이터가 골고루 포함되도록 할 수 있기 때문입니다.
    util.shuffle(data);

    // Step 2. 데이터를 텐서로 변환합니다.
    // inputs는 입력의 Example 입니다.
    // labels는 실제 출력값을 나타내는데, 이는 일종의 '정답'으로, 이를 머신러닝 상에서 `label`이라고 합니다.
    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    // 이후 각 데이터를 2D 텐서로 변환하는데,
    // Shape(두번째 인수)는 `[Example 개수, 각 Example 별 Feature 개수]`의 형태가 됩니다.
    // 현재 예제에서 Feature는 곧 하나 뿐인 `horsepower`가 됩니다.
    const inputTensor = tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tensor2d(labels, [labels.length, 1]);

    // Step 3. Min-Max 스케일링을 통해 데이터를 0 ~ 1 사이로 변환합니다.
    // ML모델 내부는 기본적으로 너무 크지 않은 숫자에 대해 적합하기 때문에 정규화가 필수적입니다.
    // 일반적인 정규화의 범위는 `0 ~ 1` 또는 `-1 ~ 1`입니다.
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // 추후에 쓰일 Min/Max의 경계를 반환합니다.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
};

const trainModel = async (
  model: Sequential,
  inputs: Tensor<Rank>,
  labels: Tensor<Rank>,
  container: HTMLElement
) => {
  // 모델을 학습하기 전에는 컴파일 과정이 필요합니다.
  model.compile({
    // tf에는 다양한 옵티마이저가 존재하며, 여기서는 별도의 세팅이 필요없는 adam 옵티마이저를 사용합니다.
    optimizer: train.adam(),
    // `loss`는 각 배치를 얼마나 잘 학습하고 있는지 모델에 알려줄 함수로,
    // 여기서 쓰인 `meanSquaredError`는 모델이 수행한 예측을 실제값과 비교하는 함수입니다.
    loss: losses.meanSquaredError,
    metrics: ["mse"],
  });

  // `batchSize`는 반복하게 될 각 학습에서 모델이 활용할 데이터 하위 집합의 크기입니다.
  // 일반적으로는 32~512 범위입니다.
  // 언제나 이상적인 배치의 크기는 없습니다.
  const batchSize = 32;
  // `epochs`는 모델이 제공된 전체 데이터 세트를 볼 횟수입니다.
  // 여기서는 데이터 셋을 50번 반복합니다.
  const epochs = 100;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: show.fitCallbacks(container, ["loss", "mse"], {
      height: 100,
      callbacks: ["onEpochEnd"],
    }),
  });
};

const testModel = async (
  model: Sequential,
  inputData: any,
  normalizationData: any,
  container: HTMLElement
) => {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;
  const [xs, preds] = tidy(() => {
    // linspace로 모델에 제공할 새 "예시"를 100개 생성합니다.
    const xs = linspace(0, 1, 100);
    // `model.predict`는 만든 예시를 모델에 전달하는 방법이며,
    // 이는 학습할 때와 유사한 Tensor 형태여야 합니다.
    const preds = model.predict(xs.reshape([100, 1])) as Tensor<Rank>;

    // 아래는 기존에 모델 내에서 취하여 "정규화"되었던 데이터를 다시 되돌리는 작업입니다.
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d: any) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  render.scatterplot(
    container,
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );
};

export default defineComponent({
  setup() {
    const plotContainer = ref<HTMLElement | null>(null);
    const summaryContainer = ref<HTMLElement | null>(null);
    const trainContainer = ref<HTMLElement | null>(null);
    const testContainer = ref<HTMLElement | null>(null);
    onMounted(async () => {
      const data = await getCarsData();
      const values = data.map(({ horsepower, mpg }) => ({
        x: horsepower,
        y: mpg,
      }));
      render.scatterplot(
        plotContainer.value!,
        { values },
        {
          xLabel: "Horsepower",
          yLabel: "MPG",
          height: 300,
        }
      );
      const model = runModel(summaryContainer.value!);
      const tensorData = convertToTensor(data);
      const { inputs, labels } = tensorData;
      await trainModel(model, inputs, labels, trainContainer.value!);
      await testModel(model, data, tensorData, testContainer.value!);
    });
    return {
      plotContainer,
      summaryContainer,
      trainContainer,
      testContainer,
    };
  },
});
</script>
