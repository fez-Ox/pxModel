package com.pxmodel.classifier

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: Classifier
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    private val LABELS = arrayOf("damaged", "plastic_wrap", "sealed", "open")

    private val pickImageContract =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { loadAndClassify(it) }
        }

    private val cameraContract =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
            bitmap?.let { classify(it) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        classifier = Classifier(this)

        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)

        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            pickImageContract.launch("image/*")
        }

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            cameraContract.launch(null)
        }
    }

    private fun loadAndClassify(uri: Uri) {
        try {
            val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
            imageView.setImageBitmap(bitmap)
            classify(bitmap)
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
        }
    }

    private fun classify(bitmap: Bitmap) {
        try {
            val probs = classifier.predict(bitmap)
            val sb = StringBuilder()
            for (i in LABELS.indices) {
                sb.appendLine(
                    String.format(Locale.US, "%s: %.1f%%", LABELS[i], probs[i] * 100)
                )
            }
            resultText.text = sb.toString()
        } catch (e: Exception) {
            resultText.text = "Error: ${e.message}"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::classifier.isInitialized) classifier.close()
    }
}
