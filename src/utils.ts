const jimp = require('jimp');
import * as tensorflow from '@tensorflow/tfjs';


/**
 * Write an image tensor to a image file.
 *
 * @param {tf.Tensor} imageTensor The image tensor to write to file.
 *   Assumed to be an int32-type tensor with value in the range 0-255.
 * @param {string} filePath Destination file path.
 */
export const writeImageTensorToFile = async (imageTensor: tensorflow.Tensor<tensorflow.Rank>, filePath: string) => {
    const imageH: number = imageTensor.shape[1] as number;
    const imageW: number = imageTensor.shape[2]  as number;
    const imageData = imageTensor.dataSync();
  
    const bufferLen = imageH * imageW * 4;
    const buffer = new Uint8Array(bufferLen);
    let index = 0;
    for (let i = 0; i < imageH; ++i) {
      for (let j = 0; j < imageW; ++j) {
        const inIndex = 3 * (i * imageW + j);
        buffer.set([Math.floor(imageData[inIndex])], index++);
        buffer.set([Math.floor(imageData[inIndex + 1])], index++);
        buffer.set([Math.floor(imageData[inIndex + 2])], index++);
        buffer.set([255], index++);
      }
    }
  
    return new Promise((resolve, reject) => {
      new jimp(
          {data: new Buffer(buffer), width: imageW, height: imageH},
          (err: any, img: any) => {
            if (err) {
              reject(err);
            } else {
              img.write(filePath);
              resolve();
            }
          });
    });
  }