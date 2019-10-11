import { createTrainingSet, createPredictionSet } from './data/dataset';
import * as tensorflow from '@tensorflow/tfjs';
import { Classifier } from '@piximi/types';
import { gradClassActivationMap } from './grad-CAM';
import { writeImageTensorToFile } from './utils';
import {MnistData} from './data/data';



const runGradCAM = async () => {
    var path = require("path");
    var fs = require("fs");
    var testFilePath = path.resolve('src', 'data', 'smallMNISTTest.piximi');
    var stringContent = fs.readFileSync(testFilePath);
    var classifier = JSON.parse(stringContent) as Classifier;

    const numberOfClasses = classifier.categories.length - 1;
    const trainingSet = await createTrainingSet(classifier.categories, classifier.images, numberOfClasses);

    const predictionSet = await createPredictionSet(classifier.images);
    
    const model = await createModel();

    model.compile({
        loss: tensorflow.losses.softmaxCrossEntropy,
        metrics: ['accuracy'],
        optimizer: tensorflow.train.adadelta(0.08)
    });
    
    const args = {
        epochs: 5,
        batchSize: 400,
        shuffle: true,
        validationSplit: 0.2,
    };

    await model.fit(trainingSet.data, trainingSet.lables, args);

    var testPred = model.predict([predictionSet.data[0]]);
    //var testPred = model.apply([predictionSet.data[0]]);

    var gradCAM = await gradClassActivationMap(model, 1, classifier.images[0]);

    writeImageTensorToFile(gradCAM, "gradCAM.png");
};

runGradCAM();

const createModel = async () => {
    const model = tensorflow.sequential();
    model.add(tensorflow.layers.conv2d({inputShape: [28,28,3], kernelSize: 3, filters: 16, activation: 'relu'}));
    model.add(tensorflow.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tensorflow.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    model.add(tensorflow.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tensorflow.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    model.add(tensorflow.layers.flatten());
    model.add(tensorflow.layers.dense({units: 32, activation: 'relu'}));
    model.add(tensorflow.layers.dense({units: 2, activation: 'relu'}));
    model.add(tensorflow.layers.dense({units: 2, activation: 'softmax'}));
    return model;
}
