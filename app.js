// app.js

let srcImage = null; // 存储原始上传的图像数据

// OpenCV.js 加载完成后的回调
function onOpenCvReady() {
  console.log('OpenCV.js is ready.');
  document.getElementById('loading').classList.add('hidden');

  // 绑定事件监听器
  setupEventListeners();
}

function setupEventListeners() {
  const cameraInput = document.getElementById('cameraInput');
  const fileInput = document.getElementById('fileInput');
  const cameraBtn = document.getElementById('cameraBtn');
  const galleryBtn = document.getElementById('galleryBtn');

  // 触发文件选择
  cameraBtn.addEventListener('click', () => cameraInput.click());
  galleryBtn.addEventListener('click', () => fileInput.click());

  // 处理图片上传
  cameraInput.addEventListener('change', handleImageUpload);
  fileInput.addEventListener('change', handleImageUpload);

  // 绑定所有参数滑块和复选框的 input/change 事件，实时更新图像
  const inputs = ['blurSize', 'minArea', 'maxArea', 'kernelSize', 'separationLevel', 'minCircularity'];
  inputs.forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', (e) => {
      // 同步更新 UI 上的数值显示
      let valId = id.replace('Size', 'Val').replace('Level', 'Val').replace('minCircularity', 'circVal');
      if (document.getElementById(valId)) {
          document.getElementById(valId).innerText = e.target.value;
      }
      if (srcImage) processImage();
    });
  });

  document.getElementById('invertToggle').addEventListener('change', () => {
    if (srcImage) processImage();
  });
}

function handleImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(event) {
    const imgElement = new Image();
    imgElement.onload = function() {
      // 限制最大处理尺寸以保证性能
      const MAX_WIDTH = 800;
      let width = imgElement.width;
      let height = imgElement.height;

      if (width > MAX_WIDTH) {
        height = Math.round((height * MAX_WIDTH) / width);
        width = MAX_WIDTH;
      }

      const canvas = document.getElementById('originalCanvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imgElement, 0, 0, width, height);

      // 如果之前有图像，先释放内存
      if (srcImage) {
        srcImage.delete();
      }

      // 读取新图像到 OpenCV 矩阵
      srcImage = cv.imread(canvas);

      document.getElementById('resultSection').classList.remove('hidden');
      processImage(); // 开始处理
    };
    imgElement.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function processImage() {
  if (!srcImage || srcImage.empty()) return;

  // 1. 获取 UI 参数
  let blurSize = parseInt(document.getElementById('blurSize').value);
  let minArea = parseInt(document.getElementById('minArea').value);
  let maxArea = parseInt(document.getElementById('maxArea').value);
  let kernelSize = parseInt(document.getElementById('kernelSize').value);
  let separationLevel = parseInt(document.getElementById('separationLevel').value) / 100.0;
  let minCircularity = parseInt(document.getElementById('minCircularity').value) / 100.0;
  let invert = document.getElementById('invertToggle').checked;

  // 确保核大小为奇数
  blurSize = blurSize % 2 === 0 ? blurSize + 1 : blurSize;
  kernelSize = kernelSize % 2 === 0 ? kernelSize + 1 : kernelSize;

  // 2. 初始化 OpenCV 矩阵 (及时 delete 防止内存泄漏)
  let gray = new cv.Mat();
  let blur = new cv.Mat();
  let thresh = new cv.Mat();
  let distTrans = new cv.Mat();
  let sureFg = new cv.Mat();
  let hierarchy = new cv.Mat();
  let contours = new cv.MatVector();
  let resultCanvas = srcImage.clone();

  try {
    // 3. 图像预处理
    cv.cvtColor(srcImage, gray, cv.COLOR_RGBA2GRAY, 0);
    cv.GaussianBlur(gray, blur, new cv.Size(blurSize, blurSize), 0, 0, cv.BORDER_DEFAULT);

    // 4. 自适应二值化 (对抗阴影和光照不均)
    if (invert) {
      cv.adaptiveThreshold(blur, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10);
    } else {
      cv.adaptiveThreshold(blur, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 10);
    }

    // 5. 形态学开运算 (去除小白噪点)
    let M = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kernelSize, kernelSize));
    cv.morphologyEx(thresh, thresh, cv.MORPH_OPEN, M);

    // 6. 距离变换 (分离粘连种子的核心)
    cv.distanceTransform(thresh, distTrans, cv.DIST_L2, 5);
    cv.normalize(distTrans, distTrans, 1, 0, cv.NORM_INF); // 归一化到 0~1 之间

    // 根据分离强度参数进行阈值切割，提取种子核心区域
    cv.threshold(distTrans, sureFg, separationLevel, 255, cv.THRESH_BINARY);
    sureFg.convertTo(sureFg, cv.CV_8U);

    // 7. 查找轮廓
    cv.findContours(sureFg, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let validCount = 0;

    // 8. 过滤与绘制
    for (let i = 0; i < contours.size(); ++i) {
      let cnt = contours.get(i);
      let area = cv.contourArea(cnt);

      // 计算圆度 (4 * pi * area / perimeter^2)
      let perimeter = cv.arcLength(cnt, true);
      let circularity = 0;
      if (perimeter > 0) {
        circularity = (4 * Math.PI * area) / (perimeter * perimeter);
      }

      // 注意：距离变换后的核心区域面积比实际种子小，因此我们将 minArea 标准降低 (例如乘以 0.2) 来适应
      if (area >= minArea * 0.2 && area <= maxArea && circularity >= minCircularity) {
        validCount++;

        // 计算轮廓的质心，用于绘制数字编号
        let M_moments = cv.moments(cnt, false);
        let cx = M_moments.m10 / M_moments.m00;
        let cy = M_moments.m01 / M_moments.m00;

        // 绘制轮廓 (绿色) 和 编号 (红色)
        cv.drawContours(resultCanvas, contours, i, new cv.Scalar(0, 255, 0, 255), 2);
        cv.putText(resultCanvas, validCount.toString(), new cv.Point(cx - 5, cy + 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, new cv.Scalar(255, 0, 0, 255), 1);
      }
    }

    // 9. 更新 UI 结果
    document.getElementById('seedCount').innerText = validCount;
    cv.imshow('resultCanvas', resultCanvas);
    cv.imshow('threshCanvas', sureFg); // 调试视图显示距离变换切割后的核心，方便调参

  } catch (err) {
    console.error("图像处理出错:", err);
  } finally {
    // 10. 释放所有 OpenCV 对象内存，极其重要！防止浏览器崩溃
    gray.delete(); blur.delete(); thresh.delete();
    distTrans.delete(); sureFg.delete();
    hierarchy.delete(); contours.delete(); resultCanvas.delete();
    if (typeof M !== 'undefined') M.delete();
  }
}
