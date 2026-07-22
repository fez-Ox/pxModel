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
    val gate: GateOption = DEFAULT_GATE,
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

    enum class GateOption(val displayName: String) {
        SIMPLE("Simple"),
        YOLO_GATE("YOLO Package Gate");

        override fun toString(): String = displayName
    }

    data class GateInfo(
        val packageDetected: Boolean,
        val confidence: Float,
    )

    data class Result(
        val probabilities: FloatArray,
        val inferenceTimeMs: Double,
        val gateInfo: GateInfo? = null,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Result) return false
            return probabilities.contentEquals(other.probabilities) &&
                inferenceTimeMs == other.inferenceTimeMs &&
                gateInfo == other.gateInfo
        }

        override fun hashCode(): Int {
            var result = probabilities.contentHashCode()
            result = 31 * result + inferenceTimeMs.hashCode()
            result = 31 * result + (gateInfo?.hashCode() ?: 0)
            return result
        }
    }

    companion object {
        private const val TAG = "Classifier"
        private const val COND_INPUT_SIZE = 224
        private const val YOLO_INPUT_SIZE = 640
        private const val YOLO_GATE_ASSET = "yolo_package_gate.tflite"
        private const val YOLO_MAX_DETECTIONS = 300
        private const val YOLO_STRIDE = 38
        private const val YOLO_CONF_IDX = 4
        private const val YOLO_CLASS_IDX = 5
        private const val YOLO_PACKAGE_CLASS_ID = 0
        private const val GATE_CONF_THRESHOLD = 0.05f

        val DEFAULT_MODEL = ModelOption.CONVNEXT_TINY
        val DEFAULT_RUNTIME = RuntimeOption.COMPILED_MODEL_CPU
        val DEFAULT_GATE = GateOption.SIMPLE
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

    private var gateInterpreter: Interpreter? = null
    private var gateMappedModel: MappedByteBuffer? = null

    val runtimeName: String
        get() = runtime.displayName

    init {
        try {
            initializeCompiledModel(context)

            if (gate == GateOption.YOLO_GATE) {
                initializeGateModel(context)
            }

            Log.i(TAG, "Loaded ${model.displayName} with $runtimeName · $gate")
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

    private fun initializeGateModel(context: Context) {
        gateMappedModel = loadModelFile(context, YOLO_GATE_ASSET)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        gateInterpreter = Interpreter(gateMappedModel, options)
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
        val gateResult = if (gate == GateOption.YOLO_GATE) {
            runGate(bitmap)
        } else {
            null
        }

        if (gateResult != null && !gateResult.packageDetected) {
            val probs = FloatArray(LABELS.size) { 0f }
            probs[LABELS.indexOf("non_package")] = 1f
            return Result(probs, gateResult.inferenceTimeMs, gateResult)
        }

        return runConditionModel(bitmap, gateResult?.inferenceTimeMs ?: 0.0)
    }

    private fun runConditionModel(bitmap: Bitmap, priorTimeMs: Double = 0.0): Result {
        val resized = Bitmap.createScaledBitmap(bitmap, COND_INPUT_SIZE, COND_INPUT_SIZE, true)
        val input = condPreprocess(resized)

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

        return if (gate == GateOption.YOLO_GATE) {
            val gateResult = GateInfo(true, 1.0f)
            Result(probabilities, priorTimeMs + elapsedMs, gateResult)
        } else {
            Result(probabilities, elapsedMs)
        }
    }

    private fun runGate(bitmap: Bitmap): GateInfo {
        val resized = Bitmap.createScaledBitmap(bitmap, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, true)
        val input = gatePreprocess(resized)

        val output = Array(1) { Array(YOLO_MAX_DETECTIONS) { FloatArray(YOLO_STRIDE) } }
        val start = System.nanoTime()

        val inputBuffer = input.toDirectByteBuffer()
        requireNotNull(gateInterpreter).run(inputBuffer, output)

        val elapsedMs = (System.nanoTime() - start) / 1_000_000.0

        var maxConf = 0.0f
        for (i in 0 until YOLO_MAX_DETECTIONS) {
            val conf = output[0][i][YOLO_CONF_IDX]
            val cls = output[0][i][YOLO_CLASS_IDX].toInt()
            if (cls == YOLO_PACKAGE_CLASS_ID && conf > GATE_CONF_THRESHOLD && conf > maxConf) {
                maxConf = conf
            }
        }

        return GateInfo(
            packageDetected = maxConf > 0f,
            confidence = maxConf,
        )
    }

    private fun condPreprocess(bitmap: Bitmap): FloatArray {
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

    private fun gatePreprocess(bitmap: Bitmap): FloatArray {
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
            values[index] = red
            values[index + planeSize] = green
            values[index + 2 * planeSize] = blue
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

        gateInterpreter?.close()
        gateInterpreter = null
        gateMappedModel = null
    }
}
