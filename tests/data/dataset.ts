import { Category, Image } from '@piximi/types';
import * as ImageJS from 'image-js';
import * as tensorflow from '@tensorflow/tfjs';

export const createDataset = async (
  categories: Category[],
  images: Image[]
) => {
  const trainData = images.filter(
    (image: Image) =>
      image.categoryIdentifier !== '00000000-0000-0000-0000-000000000000'
  );

  const dataSet = await createLabledTensorflowDataSet(
    trainData,
    categories
  );

  return { dataSet: dataSet, numberOfCategories:  categories.length - 1};
};

const findCategoryIndex = (
  categories: Category[],
  identifier: string
): number => {
  return categories.findIndex(
    (category: Category) => category.identifier === identifier
  );
};

const tensorImageData = async (image: Image) => {
  const data = await ImageJS.Image.load(image.data);

  var tensorImage = tensorflow.browser
      .fromPixels(data.getCanvas())
      .toFloat()
      .sub(tensorflow.scalar(127.5))
      .div(tensorflow.scalar(127.5))

  return tensorImage.reshape([1, 28, 28, 3]);
};

const createLabledTensorflowDataSet = async (
  labledData: Image[],
  categories: Category[]
) => {
  var dataSet = [];

  for (const image of labledData) {
    let tensorData = await tensorImageData(image);
    let lable = findCategoryIndex(categories, image.categoryIdentifier) - 1;
    dataSet.push( {data: tensorData, lables: lable} );
  }

  return dataSet;
};