package com.pxmodel.classifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private var classifier: Classifier? = null
    private lateinit var imageView: ImageView
    private lateinit var modelSpinner: Spinner
    private lateinit var runtimeSpinner: Spinner
    private lateinit var runtimeStatusText: TextView
    private lateinit var resultText: TextView
    private var selectedBitmap: Bitmap? = null
    private var selectedRuntime = Classifier.DEFAULT_RUNTIME
    private var gpuSupported = false

    private val pickImageContract =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let(::loadImage)
        }

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                cameraContract.launch(null)
            } else {
                Toast.makeText(this, R.string.camera_permission_denied, Toast.LENGTH_SHORT).show()
            }
        }

    private val cameraContract =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
            if (bitmap != null) {
                setSelectedImage(bitmap)
            } else {
                Toast.makeText(this, R.string.camera_capture_failed, Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        modelSpinner = findViewById(R.id.modelSpinner)
        runtimeSpinner = findViewById(R.id.runtimeSpinner)
        runtimeStatusText = findViewById(R.id.runtimeStatusText)
        resultText = findViewById(R.id.resultText)

        gpuSupported = Classifier.isGpuDelegateSupported()
        configureRuntimeSelector()
        configureModelSelector()

        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            pickImageContract.launch("image/*")
        }
        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA,
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                cameraContract.launch(null)
            } else {
                requestCameraPermission.launch(Manifest.permission.CAMERA)
            }
        }
        findViewById<Button>(R.id.btnInfer).setOnClickListener {
            selectedBitmap?.let(::classify)
                ?: run { resultText.text = getString(R.string.no_image) }
        }
        findViewById<Button>(R.id.btnRemove).setOnClickListener {
            selectedBitmap = null
            imageView.setImageDrawable(null)
            resultText.text = getString(R.string.results_placeholder)
        }
    }

    private fun configureRuntimeSelector() {
        val runtimes = Classifier.RuntimeOption.entries.toTypedArray()
        runtimeSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            runtimes,
        )
        runtimeSpinner.setSelection(runtimes.indexOf(Classifier.DEFAULT_RUNTIME))
        runtimeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: android.view.View?,
                position: Int,
                id: Long,
            ) {
                selectedRuntime = runtimes[position]
                reloadCurrentModel()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        runtimeStatusText.text = getString(
            if (gpuSupported) R.string.gpu_supported else R.string.gpu_unsupported,
        )
    }

    private fun configureModelSelector() {
        val models = Classifier.ModelOption.entries.toTypedArray()
        modelSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            models,
        )
        modelSpinner.setSelection(models.indexOf(Classifier.DEFAULT_MODEL))
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: android.view.View?,
                position: Int,
                id: Long,
            ) {
                switchModel(models[position])
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        switchModel(Classifier.DEFAULT_MODEL)
    }

    @Suppress("DEPRECATION")
    private fun loadImage(uri: Uri) {
        try {
            setSelectedImage(MediaStore.Images.Media.getBitmap(contentResolver, uri))
        } catch (e: Exception) {
            Toast.makeText(this, R.string.image_load_error, Toast.LENGTH_SHORT).show()
        }
    }

    private fun setSelectedImage(bitmap: Bitmap) {
        selectedBitmap = bitmap
        imageView.setImageBitmap(bitmap)
        resultText.text = getString(R.string.image_ready)
    }

    private fun reloadCurrentModel() {
        switchModel(classifier?.model ?: Classifier.DEFAULT_MODEL)
    }

    private fun switchModel(model: Classifier.ModelOption) {
        if (
            classifier?.model == model &&
            classifier?.runtime == selectedRuntime
        ) return

        classifier?.close()
        classifier = null
        try {
            classifier = Classifier(this, model, selectedRuntime)
            resultText.text = getString(
                R.string.model_runtime_ready,
                model.displayName,
                classifier?.runtimeName,
            )
        } catch (e: Exception) {
            resultText.text = getString(
                R.string.model_load_error,
                model.displayName,
                e.message ?: e.javaClass.simpleName,
            )
        }
    }

    private fun classify(bitmap: Bitmap) {
        try {
            val activeClassifier = classifier
                ?: throw IllegalStateException("No model is loaded")
            val result = activeClassifier.predict(bitmap)

            resultText.text = buildString {
                appendLine("Model: ${activeClassifier.model.displayName}")
                appendLine("Runtime: ${activeClassifier.runtimeName}")
                appendLine(
                    String.format(Locale.US, "Inference: %.3f ms", result.inferenceTimeMs),
                )
                appendLine()
                for (index in Classifier.LABELS.indices) {
                    val label = Classifier.LABELS[index].replace('_', ' ')
                    appendLine(
                        String.format(
                            Locale.US,
                            "%s  %.1f%%",
                            label,
                            result.probabilities[index] * 100,
                        ),
                    )
                }
            }.trimEnd()
        } catch (e: Exception) {
            resultText.text = getString(R.string.inference_error, e.message)
        }
    }

    override fun onDestroy() {
        classifier?.close()
        classifier = null
        super.onDestroy()
    }
}
