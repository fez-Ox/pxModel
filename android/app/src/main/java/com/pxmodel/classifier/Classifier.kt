package com.pxmodel.classifier

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Classifier(
    context: Context,
    val model: ModelOption = DEFAULT_MODEL,
    val runtime: RuntimeOption = DEFAULT_RUNTIME,
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

    enum class RuntimeOption(val displayName: String) {
        CPU_SINGLE_THREAD("Interpreter · CPU · 1 thread"),
        XNNPACK("Interpreter · XNNPACK · 4 threads"),
        GPU_DELEGATE("LiteRT CompiledModel · GPU"),
        COMPILED_MODEL("LiteRT CompiledModel · CPU"),
        ;

        override fun toString(): String = displayName
    }

    companion object {
        private const val TAG = "Classifier"
        private const val INPUT_SIZE = 224

        val DEFAULT_MODEL = ModelOption.CONVNEXT_TINY
        val DEFAULT_RUNTIME = RuntimeOption.XNNPACK
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

    private var interpreter: Interpreter? = null

    private var environment: Environment? = null
    private var compiledModel: CompiledModel? = null
    private var compiledInputs: List<TensorBuffer> = emptyList()
    private var compiledOutputs: List<TensorBuffer> = emptyList()

    val runtimeName: String
        get() = runtime.displayName

    init {
        try {
            when (runtime) {
                RuntimeOption.COMPILED_MODEL,
                RuntimeOption.GPU_DELEGATE,
                -> initializeCompiledModel(context)
                else -> initializeInterpreter(context)
            }
            Log.i(TAG, "Loaded ${model.displayName} with $runtimeName")
        } catch (e: Throwable) {
            close()
            throw e
        }
    }

    data class Result(
        val probabilities: FloatArray,
        val inferenceTimeMs: Double,
    )

    private fun initializeInterpreter(context: Context) {
        val options = Interpreter.Options()
        when (runtime) {
            RuntimeOption.CPU_SINGLE_THREAD -> {
                options.setUseXNNPACK(false)
                options.setNumThreads(1)
            }
            RuntimeOption.XNNPACK -> {
                options.setUseXNNPACK(true)
                options.setNumThreads(4)
            }
            RuntimeOption.GPU_DELEGATE,
            RuntimeOption.COMPILED_MODEL,
            -> error("CompiledModel runtimes use a separate initialization path")
        }

        interpreter = Interpreter(loadModelFile(context, model.assetPath), options).also {
            val outputShape = it.getOutputTensor(0).shape()
            require(outputShape.contentEquals(intArrayOf(1, LABELS.size))) {
                "${model.displayName} output shape ${outputShape.contentToString()} " +
                    "does not match the ${LABELS.size}-class label schema. " +
                    "Re-export the model with the non_package class."
            }
        }
    }

    private fun initializeCompiledModel(context: Context) {
        environment = Environment.create()
        val accelerator = when (runtime) {
            RuntimeOption.COMPILED_MODEL -> Accelerator.CPU
            RuntimeOption.GPU_DELEGATE -> Accelerator.GPU
            else -> error("Interpreter runtime cannot initialize CompiledModel")
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
        val input = preprocess(resized)

        val logits: FloatArray
        val start = System.nanoTime()
        if (
            runtime == RuntimeOption.COMPILED_MODEL ||
            runtime == RuntimeOption.GPU_DELEGATE
        ) {
            compiledInputs.single().writeFloat(input)
            requireNotNull(compiledModel).run(compiledInputs, compiledOutputs)
            logits = compiledOutputs.single().readFloat()
        } else {
            val inputBuffer = input.toDirectByteBuffer()
            val output = Array(1) { FloatArray(LABELS.size) }
            requireNotNull(interpreter).run(inputBuffer, output)
            logits = output[0]
        }
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

    private fun FloatArray.toDirectByteBuffer(): ByteBuffer =
        ByteBuffer.allocateDirect(size * Float.SIZE_BYTES).apply {
            order(ByteOrder.nativeOrder())
            asFloatBuffer().put(this@toDirectByteBuffer)
            rewind()
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

        interpreter?.close()
        interpreter = null
    }
}
