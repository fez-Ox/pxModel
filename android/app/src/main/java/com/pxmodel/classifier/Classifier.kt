package com.pxmodel.classifier

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer

class Classifier(
    context: Context,
    val model: ModelOption = DEFAULT_MODEL,
    val runtime: RuntimeOption = DEFAULT_RUNTIME,
) {

    enum class ModelOption(val displayName: String, val assetPath: String) {
        MOBILENET_V3_LARGE("MobileNet-V3 Large", "mobilenet_v3_large_multilabel.tflite"),
        EFFICIENTNET_B3("EfficientNet-B3", "efficientnet_b3_multilabel.tflite"),
        CONVNEXT_TINY("ConvNeXt-Tiny", "convnext_tiny_multilabel.tflite");

        override fun toString(): String = displayName
    }

    enum class RuntimeOption(val displayName: String) {
        COMPILED_MODEL_CPU("CompiledModel · CPU"),
        COMPILED_MODEL_GPU("CompiledModel · GPU");

        override fun toString(): String = displayName
    }

    data class Result(
        val probabilities: FloatArray,
        val inferenceTimeMs: Double,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Result) return false
            return probabilities.contentEquals(other.probabilities) &&
                inferenceTimeMs == other.inferenceTimeMs
        }

        override fun hashCode(): Int {
            var result = probabilities.contentHashCode()
            result = 31 * result + inferenceTimeMs.hashCode()
            return result
        }
    }

    companion object {
        private const val TAG = "Classifier"
        private const val INPUT_SIZE = 224

        val DEFAULT_MODEL = ModelOption.CONVNEXT_TINY
        val DEFAULT_RUNTIME = RuntimeOption.COMPILED_MODEL_CPU
        val LABELS = arrayOf("damaged", "plastic_wrap", "sealed", "open", "non_package")

        fun isGpuDelegateSupported(): Boolean = try {
            Environment.create().use { environment ->
                Accelerator.GPU in environment.getAvailableAccelerators()
            }
        } catch (e: Throwable) {
            Log.w(TAG, "GPU accelerator compatibility check failed", e)
            false
        }

        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private var environment: Environment? = null
    private var compiledModel: CompiledModel? = null
    private var compiledInputs: List<TensorBuffer> = emptyList()
    private var compiledOutputs: List<TensorBuffer> = emptyList()

    val runtimeName: String
        get() = runtime.displayName

    init {
        try {
            initializeCompiledModel(context)
            Log.i(TAG, "Loaded ${model.displayName} with $runtimeName")
        } catch (e: Throwable) {
            close()
            throw e
        }
    }

    private fun initializeCompiledModel(context: Context) {
        environment = Environment.create()
        val accelerator = when (runtime) {
            RuntimeOption.COMPILED_MODEL_CPU -> Accelerator.CPU
            RuntimeOption.COMPILED_MODEL_GPU -> Accelerator.GPU
        }
        val options = CompiledModel.Options(accelerator).apply {
            if (accelerator == Accelerator.CPU) {
                cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
            }
        }
        compiledModel = CompiledModel.create(
            context.assets,
            model.assetPath,
            options,
            requireNotNull(environment),
        )
        compiledInputs = requireNotNull(compiledModel).createInputBuffers()
        compiledOutputs = requireNotNull(compiledModel).createOutputBuffers()

        require(compiledInputs.size == 1 && compiledOutputs.size == 1) {
            "Expected one model input and one output, but CompiledModel exposed " +
                "${compiledInputs.size} inputs and ${compiledOutputs.size} outputs"
        }
    }

    fun predict(bitmap: Bitmap): Result {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val input = preprocess(resized)

        val logits: FloatArray
        val start = System.nanoTime()

        compiledInputs.single().writeFloat(input)
        requireNotNull(compiledModel).run(compiledInputs, compiledOutputs)
        logits = compiledOutputs.single().readFloat()

        val elapsedMs = (System.nanoTime() - start) / 1_000_000.0

        require(logits.size == LABELS.size) {
            "Runtime returned ${logits.size} values; expected ${LABELS.size}"
        }
        val probabilities = FloatArray(LABELS.size) { sigmoid(logits[it]) }
        return Result(probabilities, elapsedMs)
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
            val red = ((pixel shr 16) and 0xFF) / 255.0f
            val green = ((pixel shr 8) and 0xFF) / 255.0f
            val blue = (pixel and 0xFF) / 255.0f
            values[index] = (red - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
            values[index + planeSize] = (green - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
            values[index + 2 * planeSize] = (blue - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
            index++
        }
        return values
    }

    private fun sigmoid(value: Float): Float =
        1.0f / (1.0f + kotlin.math.exp(-value))

    fun close() {
        compiledInputs.forEach(TensorBuffer::close)
        compiledOutputs.forEach(TensorBuffer::close)
        compiledInputs = emptyList()
        compiledOutputs = emptyList()
        compiledModel?.close()
        compiledModel = null
        environment?.close()
        environment = null
    }
}
