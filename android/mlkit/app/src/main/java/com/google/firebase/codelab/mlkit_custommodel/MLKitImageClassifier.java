/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.firebase.codelab.mlkit_custommodel;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.support.annotation.NonNull;
import android.text.TextUtils;
import android.util.Log;
import android.widget.Toast;

import com.google.android.gms.tasks.Continuation;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseModelOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.google.firebase.ml.custom.model.FirebaseCloudModelSource;
import com.google.firebase.ml.custom.model.FirebaseLocalModelSource;
import com.google.firebase.ml.custom.model.FirebaseModelDownloadConditions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutionException;

/**
 * Classifies images with Tensorflow Lite.
 */
public class MLKitImageClassifier {
    public static final String TAG = MLKitImageClassifier.class.getSimpleName();
    /**
     * Name of the model file hosted with Firebase.
     */
    private static final String HOSTED_MODEL_NAME = "mobilenet_v1_224";
    private static final String LOCAL_MODEL_ASSET = "mobilenet_v1.0_224.tflite";
    /**
     * Name of the label file stored in Assets.
     */
    private static final String LABEL_PATH = "labels.txt";
    /**
     * Number of results to show in the UI.
     */
    private static final int RESULTS_TO_SHOW = 3;
    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    static final int DIM_IMG_SIZE_X = 224;
    static final int DIM_IMG_SIZE_Y = 224;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    /* Preallocated buffers for storing image data in. */
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    ByteBuffer imgData = null;

    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> labelList;

    private final PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float>
                                o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    /**
     * An instance of the driver class to run model inference with Firebase.
     */
    private FirebaseModelInterpreter mInterpreter;
    /**
     * Data configuration of input & output data of model.
     */
    private FirebaseModelInputOutputOptions mDataOptions;


    /**
     * Initializes an {@code ImageClassifier}.
     */
    MLKitImageClassifier(Activity activity) throws IOException {
        labelList = loadLabelList(activity);

        // putFloat writes 4 bytes; if using ints, only 1 byte is written
        imgData =
                ByteBuffer.allocateDirect(
                        4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());

        int[] inputDims = {DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE};
        int[] outputDims = {DIM_BATCH_SIZE, labelList.size()};
        try {
            mDataOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, inputDims)
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, outputDims)
                            .build();
            FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions
                    .Builder()
                    .requireWifi()
                    .build();
            FirebaseLocalModelSource localModelSource =
                    new FirebaseLocalModelSource.Builder("asset")
                            .setAssetFilePath(LOCAL_MODEL_ASSET).build();

            FirebaseCloudModelSource cloudSource = new FirebaseCloudModelSource.Builder
                    (HOSTED_MODEL_NAME)
                    .enableModelUpdates(true)
                    .setInitialDownloadConditions(conditions)
                    .setUpdatesDownloadConditions(conditions)  // You could also specify
                    // different conditions
                    // for updates
                    .build();
            FirebaseModelManager manager = FirebaseModelManager.getInstance();
            manager.registerLocalModelSource(localModelSource);
            manager.registerCloudModelSource(cloudSource);
            FirebaseModelOptions modelOptions =
                    new FirebaseModelOptions.Builder()
                            .setCloudModelName(HOSTED_MODEL_NAME)
                            .setLocalModelName("asset")
                            .build();
            mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions);
        } catch (FirebaseMLException e) {
            showToast("Error while setting up the model", activity.getApplicationContext());
            e.printStackTrace();
        }
    }

    private void showToast(String message, Context appContext) {
        Toast.makeText(appContext, message, Toast.LENGTH_SHORT).show();
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Activity activity) {
        List<String> labelList = new ArrayList<>();
        try (BufferedReader reader =
                     new BufferedReader(new InputStreamReader(activity.getAssets().open
                             (LABEL_PATH)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to read label list.", e);
        }
        return labelList;
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private synchronized void convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        imgData.rewind();
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0,
                scaledBitmap.getWidth(), scaledBitmap.getHeight());
        // Convert the image to int points.
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
    }

    /**
     * Classifies a frame from the preview stream, returning a multiline string with results.
     */
    String classifyFrame(Bitmap bitmap) {
        if (mInterpreter == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            return "Interpreter not initialized!";
        }
        // Create input data, writing to imgData
        convertBitmapToFloatByteBuffer(bitmap);

        try {
            FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();
            // Here's where the magic happens!!
            long startTime = SystemClock.uptimeMillis();
            List<String> labels = Tasks.await(
                    mInterpreter
                            .run(inputs, mDataOptions)
                            .addOnFailureListener(new OnFailureListener() {
                                @Override
                                public void onFailure(@NonNull Exception e) {
                                    Log.e(TAG, "Error running model inference", e);
                                }
                            })
                            .continueWith(
                                    new Continuation<FirebaseModelOutputs, List<String>>() {
                                        @Override
                                        public List<String> then(Task<FirebaseModelOutputs> task) {
                                            float[][] labelProbArray = task.getResult()
                                                    .<float[][]>getOutput(0);
                                            List<String> topLabels = getTopLabels(labelProbArray);
                                            return topLabels;
                                        }
                                    })
            );
            long endTime = SystemClock.uptimeMillis();
            String inferenceTime = Long.toString(endTime - startTime) + "ms";

            return inferenceTime + "\n" + TextUtils.join("\n", labels);
        } catch (FirebaseMLException e) {
            return "Error running model inference: Firebase: " + e.getMessage();
        } catch (InterruptedException e) {
            return "Error running model inference: interrupted: " + e.getMessage();
        } catch (ExecutionException e) {
            return "Error running model inference: execution: " + e.getMessage();
        }
    }

    private synchronized List<String> getTopLabels(float[][] labelProbArray) {
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        List<String> result = new ArrayList<>();
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            result.add(0, String.format("%s: %4.2f",label.getKey(),label.getValue()));
        }
//        Log.d(TAG, "labels: " + result.toString());
        return result;
    }

    /**
     * Closes tflite to release resources.
     */
    public void close() {
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private synchronized void convertBitmapToFloatByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }
}
