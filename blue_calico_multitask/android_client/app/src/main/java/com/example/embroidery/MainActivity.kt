package com.example.embroidery

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.FileProvider
import com.example.embroidery.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var currentPhotoUri: Uri? = null
    private var currentPhotoFile: File? = null
    private val client = OkHttpClient()

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            currentPhotoUri = it
            currentPhotoFile = null
            showPreview(it)
        }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success) {
            currentPhotoUri?.let { showPreview(it) }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnCamera.setOnClickListener { openCamera() }
        binding.btnGallery.setOnClickListener { galleryLauncher.launch("image/*") }
        binding.btnPredict.setOnClickListener { predict() }
    }

    private fun openCamera() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
            return
        }
        val photoFile = createImageFile()
        currentPhotoFile = photoFile
        val uri = FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            photoFile
        )
        currentPhotoUri = uri
        cameraLauncher.launch(uri)
    }

    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.CHINA).format(Date())
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir)
    }

    private fun showPreview(uri: Uri) {
        contentResolver.openInputStream(uri)?.use {
            val bitmap = BitmapFactory.decodeStream(it)
            binding.ivPreview.setImageBitmap(bitmap)
        }
    }

    private fun predict() {
        val uri = currentPhotoUri
        if (uri == null) {
            Toast.makeText(this, "请先拍照或选择图片", Toast.LENGTH_SHORT).show()
            return
        }
        val server = binding.etServer.text.toString().trim()
        if (server.isEmpty()) {
            Toast.makeText(this, "请填写服务器地址", Toast.LENGTH_SHORT).show()
            return
        }

        binding.progressBar.visibility = View.VISIBLE
        binding.btnPredict.isEnabled = false
        binding.tvResult.text = "正在识别，请稍候..."

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = uploadImage(uri, server)
                withContext(Dispatchers.Main) {
                    binding.tvResult.text = formatResult(result)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.tvResult.text = "识别失败：${e.message}\n请确保手机与电脑在同一 Wi-Fi，且服务器已启动。"
                }
            } finally {
                withContext(Dispatchers.Main) {
                    binding.progressBar.visibility = View.GONE
                    binding.btnPredict.isEnabled = true
                }
            }
        }
    }

    private fun uploadImage(uri: Uri, server: String): JSONObject {
        val file = currentPhotoFile ?: copyUriToTempFile(uri)
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", file.name,
                file.asRequestBody("image/jpeg".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder()
            .url("http://$server/predict")
            .post(body)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("服务器错误：${response.code}")
            }
            val json = response.body?.string() ?: throw IOException("空响应")
            return JSONObject(json)
        }
    }

    private fun copyUriToTempFile(uri: Uri): File {
        val tempFile = File(cacheDir, "selected_image.jpg")
        contentResolver.openInputStream(uri)?.use { input ->
            tempFile.outputStream().use { output -> input.copyTo(output) }
        }
        return tempFile
    }

    private fun formatResult(json: JSONObject): String {
        val auth = json.getJSONObject("auth")
        val isReal = json.getBoolean("is_real")
        val sb = StringBuilder()
        sb.append("真伪：${auth.getString("name")}\n")
        sb.append("  置信度：${(auth.getDouble("prob") * 100).format()}%\n")
        sb.append("  真品 ${(auth.getJSONObject("probs").getDouble("真品 / 手工") * 100).format()}% | ")
        sb.append("伪作 ${(auth.getJSONObject("probs").getDouble("伪作 / 机绣") * 100).format()}%\n\n")

        if (isReal) {
            val pattern = json.getJSONObject("pattern")
            val defect = json.getJSONObject("defect")
            sb.append("纹样：${pattern.getString("name")}\n")
            sb.append("  置信度：${(pattern.getDouble("prob") * 100).format()}%\n\n")
            sb.append("疵点：${defect.getString("name")}\n")
            sb.append("  置信度：${(defect.getDouble("prob") * 100).format()}%\n")
        } else {
            sb.append("伪品：跳过纹样分类与疵点检测。")
        }
        return sb.toString()
    }

    private fun Double.format(): String = String.format("%.2f", this)
}
