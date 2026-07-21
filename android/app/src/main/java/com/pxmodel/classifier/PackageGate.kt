package com.pxmodel.classifier

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max

class PackageGate(
    context: Context,
    runtime: Classifier.RuntimeOption,
    private val assetPath: String = ASSET_PATH,
    private val confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
) {
    companion object {
        const val ASSET_PATH = "yolo_package_gate.tflite"
        const val INPUT_SIZE = 640
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.25f
    }

    data class Result(
        val hasPackage: Boolean,
        val confidence: Float,
        val inferenceTimeMs: Double,
    )

    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply {
            when (runtime) {
                Classifier.RuntimeOption.CPU_SINGLE_THREAD -> {
                    setUseXNNPACK(false)
                    setNumThreads(1)
                }
                else -> {
                    setUseXNNPACK(true)
                    setNumThreads(4)
                }
            }
        }
        interpreter = Interpreter(loadModelFile(context, assetPath), options)
    }

    fun predict(bitmap: Bitmap): Result {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val input = preprocess(resized).toDirectByteBuffer()
        val outputs = allocateOutputs()

        val start = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000.0

        val confidence = maxDetectionConfidence(outputs)
        return Result(confidence >= confidenceThreshold, confidence, elapsedMs)
    }

    private fun allocateOutputs(): MutableMap<Int, Any> {
        val outputs = mutableMapOf<Int, Any>()
        for (index in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(index).shape()
            outputs[index] = allocateFloatArray(shape)
        }
        return outputs
    }

    private fun allocateFloatArray(shape: IntArray): Any = when (shape.size) {
        1 -> FloatArray(shape[0])
        2 -> Array(shape[0]) { FloatArray(shape[1]) }
        3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
        4 -> Array(shape[0]) { Array(shape[1]) { Array(shape[2]) { FloatArray(shape[3]) } } }
        else -> error("Unsupported YOLO output rank: ${shape.contentToString()}")
    }

    private fun maxDetectionConfidence(outputs: Map<Int, Any>): Float {
        var best = 0f
        for (value in outputs.values) {
            best = max(best, maxConfidenceFromOutput(value))
        }
        return best
    }

    @Suppress("UNCHECKED_CAST")
    private fun maxConfidenceFromOutput(output: Any): Float = when (output) {
        is Array<*> -> {
            val first = output.firstOrNull()
            when (first) {
                is FloatArray -> maxFrom2d(output as Array<FloatArray>)
                is Array<*> -> {
                    val firstNested = first.firstOrNull()
                    if (firstNested is FloatArray) {
                        maxFrom3d(output as Array<Array<FloatArray>>)
                    } else {
                        0f
                    }
                }
                else -> 0f
            }
        }
        is FloatArray -> output.maxOrNull() ?: 0f
        else -> 0f
    }

    private fun maxFrom2d(output: Array<FloatArray>): Float {
        var best = 0f
        for (row in output) {
            if (row.size > 4) best = max(best, row[4])
        }
        return best
    }

    private fun maxFrom3d(output: Array<Array<FloatArray>>): Float {
        val tensor = output.firstOrNull() ?: return 0f
        if (tensor.isEmpty()) return 0f
        val dim1 = tensor.size
        val dim2 = tensor[0].size
        var best = 0f

        // Ultralytics TFLite exports are commonly [1, 4 + classes + masks, boxes]
        // for segmentation. For a one-class gate, class confidence is channel 4.
        if (dim1 <= dim2 && dim1 > 4) {
            for (box in 0 until dim2) best = max(best, tensor[4][box])
        }

        // Some runtimes transpose to [1, boxes, 4 + classes + masks].
        if (dim1 > dim2 && dim2 > 4) {
            for (box in 0 until dim1) best = max(best, tensor[box][4])
        }
        return best
    }

    private fun preprocess(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val planeSize = width * height
        val values = FloatArray(planeSize * 3)
        var index = 0
        for (pixel in pixels) {
            values[index] = ((pixel shr 16) and 0xFF) / 255.0f
            values[index + planeSize] = ((pixel shr 8) and 0xFF) / 255.0f
            values[index + 2 * planeSize] = (pixel and 0xFF) / 255.0f
            index++
        }
        return values
    }

    private fun FloatArray.toDirectByteBuffer(): ByteBuffer =
        ByteBuffer.allocateDirect(size * Float.SIZE_BYTES).apply {
            order(ByteOrder.nativeOrder())
            asFloatBuffer().put(this@toDirectByteBuffer)
            rewind()
        }

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

    fun close() {
        interpreter.close()
    }
}
