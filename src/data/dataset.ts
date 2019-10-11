import { Category, Image } from '@piximi/types';
import * as ImageJS from 'image-js';
import * as tensorflow from '@tensorflow/tfjs';

export const createTrainingSet = async (
  categories: Category[],
  labledData: Image[],
  numberOfClasses: number
) => {
  // const trainingData: Image[] = [];
  // for (let i = 0; i < labledData.length; i++) {
  //   if (labledData[i].partition === 0) {
  //     trainingData.push(labledData[i]);
  //   }
  // }

  const trainDataSet = await createLabledTensorflowDataSet(
    labledData,
    categories
  );

  let concatenatedTensorData = tensorflow.tidy(() =>
    tensorflow.concat(trainDataSet.data)
  );
  let concatenatedLableData = tensorflow.tidy(() =>
    tensorflow.oneHot(trainDataSet.lables, numberOfClasses)
  );

  trainDataSet.data.forEach( (tensor: tensorflow.Tensor<tensorflow.Rank>) => tensor.dispose());

  return { data: concatenatedTensorData, lables: concatenatedLableData };
};

export const createTestSet = async (
  categories: Category[],
  images: Image[]
) => {
  const labledData = images.filter((image: Image) => {
    return image.categoryIdentifier !== '00000000-0000-0000-0000-000000000000';
  });

  const testData: Image[] = [];
  for (let i = 0; i < labledData.length; i++) {
    if (labledData[i].partition === 2) {
      testData.push(labledData[i]);
    }
  }

  const testDataSet = await createLabledTensorflowDataSet(testData, categories);

  return { data: testDataSet.data, lables: testDataSet.lables };
};

export const createPredictionSet = async (images: Image[]) => {
  const predictionImageSet = images.filter(
    (image: Image) =>
      image.categoryIdentifier === '00000000-0000-0000-0000-000000000000'
  );

  const predictionTensorSet: tensorflow.Tensor<tensorflow.Rank>[] = [];
  const imageIdentifiers: string[] = [];

  for (const image of predictionImageSet) {
    predictionTensorSet.push(await tensorImageData(image));
    imageIdentifiers.push(image.identifier);
  }
  return { data: predictionTensorSet, identifiers: imageIdentifiers };
};

var TESTSET_RATIO = 0.2;

export const assignToSet = (): number => {
  const rdn = Math.random();
  if (rdn < TESTSET_RATIO) {
    return 2;
  } else {
    return 0;
  }
};

const findCategoryIndex = (
  categories: Category[],
  identifier: string
): number => {
  return categories.findIndex(
    (category: Category) => category.identifier === identifier
  );
};

export const tensorImageData = async (image: Image) => {
  const data = await ImageJS.Image.load(image.data);

  var tensorImage = tensorflow.browser
      .fromPixels(data.getCanvas())

  return tensorImage.reshape([1, 28, 28, 3]) as tensorflow.Tensor<tensorflow.Rank>;
};

const createLabledTensorflowDataSet = async (
  labledData: Image[],
  categories: Category[]
) => {
  var dataSet: tensorflow.Tensor<tensorflow.Rank>[] = [];
  var labels: number[] = [];

  for (const image of labledData) {
    let tensorData = await tensorImageData(image);
    let label = findCategoryIndex(categories, image.categoryIdentifier) - 1;
    dataSet.push(tensorData);
    labels.push(label);
  }

  return {data: dataSet, lables: labels};
};