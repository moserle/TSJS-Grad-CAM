import * as tensorflow from '@tensorflow/tfjs';
import { Image } from '@piximi/types';
import { applyColorMap } from './colorMap';
import { tensorImageData } from './data/dataset';

export async function gradClassActivationMap(
  model: tensorflow.LayersModel,
  classLable: number,
  inputImage: Image
) {
  const imageTensorData = await tensorImageData(inputImage);

  // Locate the last conv layer of the model
  let lastConvLayerIndex = model.layers.length - 1;
  while (lastConvLayerIndex >= 0) {
    if (model.layers[lastConvLayerIndex].getClassName().startsWith('Conv')) {
      break;
    }
    lastConvLayerIndex--;
  }
  const lastConvLayer = model.layers[lastConvLayerIndex];

  // Create "sub-model 1", which goes from the original input to the output of the last convolutional layer
  const lastConvLayerOutput: tensorflow.SymbolicTensor = lastConvLayer.output as tensorflow.SymbolicTensor;
  const subModel1 = tensorflow.model({
    inputs: model.inputs,
    outputs: lastConvLayerOutput
  });

  // Create "sub-model 2", which goes from the output of the last convolutional layer to the output before the softmax layer
  const subModel2Input: tensorflow.SymbolicTensor = tensorflow.input({
    shape: lastConvLayerOutput.shape.slice(1)
  });
  lastConvLayerIndex++;
  let currentOutput: tensorflow.SymbolicTensor = subModel2Input;
  while (lastConvLayerIndex < model.layers.length) {
    if (
      model.layers[lastConvLayerIndex].getConfig()['activation'] === 'softmax'
    ) {
      break;
    }
    currentOutput = model.layers[lastConvLayerIndex].apply(
      currentOutput
    ) as tensorflow.SymbolicTensor;
    lastConvLayerIndex++;
  }
  const subModel2 = tensorflow.model({
    inputs: subModel2Input,
    outputs: currentOutput
  });

  // test sub model 3
  // const subModel3 = tensorflow.model({
  //   inputs: model.inputs,
  //   outputs: currentOutput
  // });

  // Generate the heatMap
  return tensorflow.tidy(() => {
    // run the sub-model 2 and extract the slice of the probability output that corresponds to the desired class
    // @ts-ignore
    //const convOutput2ClassOutput = (input: any) => subModel2.apply(input, { training: true }).gather([classLable], 1);
    const convOutput2ClassOutput = (input: any) => model.apply(input).gather([classLable], 1);

    // This is the gradient function of the output corresponding to the desired class with respect to its input (i.e., the output of the last convolutional layer of the original model)
    const gradFunction = tensorflow.grad(convOutput2ClassOutput);

    // Calculate the values of the last conv layer's output
    const lastConvLayerOutputValues: tensorflow.Tensor<
      tensorflow.Rank
    > = subModel1.apply(imageTensorData) as tensorflow.Tensor<tensorflow.Rank>;

    // Calculate the values of gradients of the class output w.r.t. the output of the last convolutional layer
    const gradValues = gradFunction(lastConvLayerOutputValues);

    // Calculate the weights of the feature maps
    const weights = tensorflow.mean(gradValues, [0, 1, 2]);

    const weightedFeatures = lastConvLayerOutputValues.mul(weights);

    // apply ReLu to the weighted features
    var heatMap: tensorflow.Tensor<tensorflow.Rank> = weightedFeatures.relu();

    // normalize the heat map
    heatMap = heatMap.div(heatMap.max());

    // Up-sample the heat map to the size of the input image
    // @ts-ignore
    heatMap = tensorflow.image.resizeBilinear(heatMap, [
      imageTensorData.shape[1],
      imageTensorData.shape[2]
    ]);

    // Apply an RGB colormap on the heatMap to convert to grey scale heatmap into a RGB one
    var gradCAM = applyColorMap(heatMap);

    // To form the final output, overlay the color heat map on the input image
    gradCAM = gradCAM.mul(2).add(imageTensorData.div(255));

    gradCAM = gradCAM.div(gradCAM.max()).mul(255);

    // @ts-ignore
    return gradCAM = tensorflow.image.resizeBilinear(gradCAM, [
      200,
      200
    ]);
  });
}
