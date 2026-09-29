document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const classFilter = document.getElementById('class-filter');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const displayContainer = document.getElementById('display-container');
    const resultImg = document.getElementById('result-img');
    const btnSample = document.getElementById('btn-sample');
    const btnDownload = document.getElementById('btn-download');
    const webcamFeed = document.getElementById('webcam-feed');
    const statTotal = document.getElementById('stat-total');
    const statTime = document.getElementById('stat-time');
    const tableBody = document.getElementById('table-body');

    let classChart = null;

    // 1. Initialize Chart.js
    const ctx = document.getElementById('classChart').getContext('2d');
    classChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#00f0ff', '#00ff88', '#9d4edd', '#ff9e00', '#ff0055',
                    '#7000ff', '#00b4d8', '#38b000', '#ccff00', '#ff5400'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#8b9bb4', font: { family: 'Outfit', size: 11 } }
                }
            }
        }
    });

    // 2. Slider threshold listener
    confSlider.addEventListener('input', (e) => {
        confVal.textContent = `${e.target.value}%`;
    });

    // 3. Tab Switcher
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            
            const targetTab = btn.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');

            if (targetTab === 'tab-webcam') {
                webcamFeed.src = webcamFeed.getAttribute('data-src');
            } else {
                webcamFeed.src = '';
            }
        });
    });

    // 4. File Drag & Drop Handling
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) processImageFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) processImageFile(e.target.files[0]);
    });

    // 5. Send Image for Object Detection via REST API
    function processImageFile(file) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('conf_threshold', (parseFloat(confSlider.value) / 100).toFixed(2));
        formData.append('allowed_classes', classFilter.value.trim());

        // Show loading state
        statTotal.textContent = '...';
        statTime.textContent = '...';

        fetch('/api/detect-image', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                renderResults(data);
            } else {
                alert('Detection Error: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(err => {
            console.error('API Call Failed:', err);
            alert('Error connecting to Object Detection server.');
        });
    }

    // 6. Synthetic Sample Runner
    btnSample.addEventListener('click', () => {
        statTotal.textContent = '...';
        fetch('/api/detect-sample')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                renderResults(data);
            }
        });
    });

    // 7. Render Output Results (Image, Chart, Table)
    function renderResults(data) {
        resultImg.src = data.image;
        displayContainer.style.display = 'block';

        statTotal.textContent = data.total_objects;
        statTime.textContent = `${data.processing_time_ms || 25} ms`;

        // Update Table
        tableBody.innerHTML = '';
        if (data.detections.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="4" class="empty-msg">No objects detected above ${confSlider.value}% confidence threshold.</td></tr>`;
        } else {
            const classCounts = {};
            data.detections.forEach((det, idx) => {
                classCounts[det.label] = (classCounts[det.label] || 0) + 1;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${idx + 1}</td>
                    <td><span class="badge-label">${det.label.toUpperCase()}</span></td>
                    <td>${(det.confidence * 100).toFixed(1)}%</td>
                    <td>[${det.bbox.join(', ')}]</td>
                `;
                tableBody.appendChild(row);
            });

            // Update Chart.js
            classChart.data.labels = Object.keys(classCounts);
            classChart.data.datasets[0].data = Object.values(classCounts);
            classChart.update();
        }
    }

    // 8. Download Result Image
    btnDownload.addEventListener('click', () => {
        if (resultImg.src) {
            const a = document.createElement('a');
            a.href = resultImg.src;
            a.download = `object_detection_result_${Date.now()}.jpg`;
            a.click();
        }
    });
});
