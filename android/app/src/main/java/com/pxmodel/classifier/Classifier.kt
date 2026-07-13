package com.pxmodel.classifier

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Classifier(
    context: Context,
    val model: ModelOption = DEFAULT_MODEL,
    val xnnpackEnabled: Boolean = true,
    val gpuEnabled: Boolean = false,
) {

    enum class ModelOption(val displayName: String, val assetPath: String) {
        MOBILENET_V3_LARGE("MobileNet-V3 Large", "mobilenet_v3_large_multilabel.tflite"),
        EFFICIENTNET_B0("EfficientNet-B0", "efficientnet_b0_multilabel.tflite"),
        EFFICIENTNET_B3("EfficientNet-B3", "efficientnet_b3_multilabel.tflite"),
        CONVNEXT_TINY("ConvNeXt-Tiny", "convnext_tiny_multilabel.tflite"),
        CONVNEXT_BASE("ConvNeXt-Base", "convnext_base_multilabel.tflite"),
        CONVNEXT_BASE_INT8("ConvNeXt-Base · INT8", "convnext_base_int8_multilabel.tflite");

        override fun toString(): String = displayName
    }

    companion object {
        private const val TAG = "Classifier"
        private const val INPUT_SIZE = 224

        val DEFAULT_MODEL = ModelOption.CONVNEXT_TINY
        val LABELS = arrayOf("damaged", "plastic_wrap", "sealed", "open", "non_package")

        fun isGpuDelegateSupported(): Boolean = try {
            CompatibilityList().use { it.isDelegateSupportedOnThisDevice }
        } catch (e: Throwable) {
            Log.w(TAG, "GPU compatibility check failed", e)
            false
        }

        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?

    val runtimeName: String
        get() {
            val cpuRuntime = if (xnnpackEnabled) "XNNPACK · 4 threads" else "CPU · 1 thread"
            return if (gpuEnabled) "GPU delegate · $cpuRuntime fallback" else cpuRuntime
        }

    init {
        val options = Interpreter.Options().apply {
            setUseXNNPACK(xnnpackEnabled)
            setNumThreads(if (xnnpackEnabled) 4 else 1)
        }
        gpuDelegate = if (gpuEnabled) {
            CompatibilityList().use { compatibility ->
                require(compatibility.isDelegateSupportedOnThisDevice) {
                    "GPU delegate is not supported on this device"
                }
                GpuDelegate(compatibility.bestOptionsForThisDevice).also(options::addDelegate)
            }
        } else {
            null
        }
        interpreter = try {
            Interpreter(loadModelFile(context, model.assetPath), options).let { loaded ->
                try {
                    val outputShape = loaded.getOutputTensor(0).shape()
                    require(outputShape.contentEquals(intArrayOf(1, LABELS.size))) {
                        "${model.displayName} output shape ${outputShape.contentToString()} " +
                            "does not match the ${LABELS.size}-class label schema. " +
                            "Re-export the model with the non_package class."
                    }
                    loaded
                } catch (e: Throwable) {
                    loaded.close()
                    throw e
                }
            }
        } catch (e: Throwable) {
            gpuDelegate?.close()
            throw e
        }
        Log.i(
            TAG,
            "Loaded ${model.displayName} with $runtimeName; " +
                "input=${interpreter.getInputTensor(0).shape().contentToString()}, " +
                "output=${interpreter.getOutputTensor(0).shape().contentToString()}",
        )
    }

    data class Result(
        val probabilities: FloatArray,
        val inferenceTimeMs: Long,
    )

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).use { input ->
            return input.channel.map(
                FileChannel.MapMode.READ_ONLY,
                afd.startOffset,
                afd.declaredLength,
            )
        }
    }

    fun predict(bitmap: Bitmap): Result {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = preprocess(resized)
        val output = Array(1) { FloatArray(LABELS.size) }

        val start = System.nanoTime()
        interpreter.run(inputBuffer, output)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000

        val probabilities = FloatArray(LABELS.size) { sigmoid(output[0][it]) }
        return Result(probabilities, elapsedMs)
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val planeSize = width * height
        val floatArray = FloatArray(planeSize * 3)
        var index = 0
        for (pixel in pixels) {
            val red = ((pixel shr 16) and 0xFF) / 255.0f
            val green = ((pixel shr 8) and 0xFF) / 255.0f
            val blue = (pixel and 0xFF) / 255.0f
            floatArray[index] = (red - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            floatArray[index + planeSize] = (green - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            floatArray[index + 2 * planeSize] = (blue - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            index++
        }

        return ByteBuffer.allocateDirect(floatArray.size * Float.SIZE_BYTES).apply {
            order(ByteOrder.nativeOrder())
            asFloatBuffer().put(floatArray)
            rewind()
        }
    }

    private fun sigmoid(value: Float): Float =
        1.0f / (1.0f + kotlin.math.exp(-value))

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
