import { createDataset } from './data/dataset';
import * as tensorflow from '@tensorflow/tfjs';
import { Classifier } from '@piximi/types';

const runExample = async () => {
    var path = require("path");
    var fs = require("fs");
    var testFilePath = path.resolve('tests', 'data', 'smallMNISTTest.piximi');
    var stringContent = fs.readFileSync(testFilePath);
    var classifier = JSON.parse(stringContent) as Classifier;

    const dataset = await createDataset(classifier.categories, classifier.images);
    
    const testModel = await createModel();

};

runExample();

const createModel = async () => {
    const model = tensorflow.sequential();
    model.add(tensorflow.layers.conv2d({inputShape: [28,28,3], kernelSize: 3, filters: 16, activation: 'relu'}));
    model.add(tensorflow.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tensorflow.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    model.add(tensorflow.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tensorflow.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    model.add(tensorflow.layers.flatten());
    model.add(tensorflow.layers.dense({units: 64, activation: 'relu'}));
    model.add(tensorflow.layers.dense({units: 2, activation: 'softmax'}));
    return model;
}
