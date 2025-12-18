# 🌍 All-India Air Pollution Health Risk Dashboard

Professional air quality monitoring and health risk prediction system with explainable AI, policy simulation, and real-time alerts.

---

## 📱 **Device Compatibility**

This dashboard is **fully responsive** and works on all devices:

- ✅ **Mobile Phones** (320px - 640px) - Touch-optimized interface
- ✅ **Tablets** (641px - 1024px) - Balanced layout
- ✅ **Laptops/PCs** (1024px+) - Full feature display

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.10+
- pip (Python package manager)

### **Installation**

1. **Clone/Download the project:**
```bash
cd Air_Pollution_Health_Risk_Project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard:**
```bash
streamlit run streamlit_app.py
```

4. **Open in browser:**
- Local: `http://localhost:8503`
- Network: `http://<your-ip>:8503`

---

## 📊 **Dashboard Features (14 Pages)**

### **Core Analysis Pages**
| Page | Purpose | Devices |
|------|---------|---------|
| 📊 **Overview** | National AQI snapshot, KPIs | ✅ All |
| 🏙️ **City Deep-Dive** | City-wise analysis & trends | ✅ All |
| 🗺️ **State Comparison** | State-to-state benchmarking | ✅ All |
| 📈 **Analysis** | Trends, correlations, seasonal | ✅ All |
| 💔 **Health Impact** | Respiratory, asthma analysis | ✅ All |

### **Predictive & Policy Pages**
| Page | Purpose | Devices |
|------|---------|---------|
| 🎮 **Policy Simulator** | What-if scenarios | ✅ All |
| 🔮 **Prediction** | ML-based forecasts | ✅ All |
| 🤖 **Model Performance** | Accuracy & metrics | ✅ All |
| 🔍 **Explainability** | SHAP-style explanations | ✅ All |

### **Advanced Pages**
| Page | Purpose | Devices |
|------|---------|---------|
| 🚨 **Early Warning** | Real-time alerts | ✅ All |
| 📊 **Data Quality** | Confidence & reliability | ✅ Mobile+ |
| 💰 **Policy Impact** | Cost-benefit analysis | ✅ Tablet+ |
| 📥 **Reports** | CSV/PDF exports | ✅ All |
| 📋 **Executive Summary** | KPI benchmarks | ✅ All |

---

## 📱 **Mobile Experience (Phone)**

When opened on phone:
- ✅ Sidebar collapses to hamburger menu
- ✅ Single-column layout automatically
- ✅ Buttons full-width for easy tapping
- ✅ Charts responsive & zoomable
- ✅ Filters optimized for touch
- ✅ Metrics scaled for readability

**Access on Phone:**
1. Find your laptop/PC IP: `ipconfig` (Windows) / `ifconfig` (Mac/Linux)
2. On phone, visit: `http://<laptop-ip>:8503`
3. Dashboard automatically adjusts to phone screen

---

## 💻 **Desktop Experience (Laptop/PC)**

When opened on desktop:
- ✅ Full sidebar navigation
- ✅ Multi-column layouts (2-4 columns)
- ✅ All advanced features visible
- ✅ Detailed tables with horizontal scroll
- ✅ Large, interactive charts
- ✅ Side-by-side comparisons

---

## 🎮 **Usage Guide**

### **For Mobile Users:**
1. **Tap Sidebar** to navigate pages
2. **Scroll down** to see all content
3. **Tap filters** to customize data
4. **Swipe charts** to explore
5. **Long-press** for menu options

### **For Tablet Users:**
1. Use **sidebar** on left (visible if screen wide enough)
2. Tap **hamburger** (☰) if sidebar hidden
3. Enjoy **2-3 column** layouts
4. Export data using **download buttons**

### **For Desktop Users:**
1. Browse **full navigation** in sidebar
2. Use **advanced filters** for detailed analysis
3. Compare **side-by-side** visualizations
4. Generate **comprehensive reports**
5. Deep-dive into **technical sections**

---

## 🔧 **Configuration**

### **Change Default Port (if 8503 is busy):**
```bash
streamlit run streamlit_app.py --server.port 8504
```

### **Enable Public Sharing:**
```bash
streamlit run streamlit_app.py --logger.level=debug
```

### **Adjust Responsiveness:**
Edit `responsive_css` in `streamlit_app.py` to customize breakpoints.

---

## 📊 **Data**

- **Dataset:** `air_pollution_50000_rows.csv`
- **Records:** 50,000+ air quality observations
- **Features:** PM2.5, PM10, NO2, SO2, CO, AQI, health metrics
- **Coverage:** All-India (states & cities)
- **Time Period:** Multi-year historical data

---

## 🤖 **Machine Learning**

- **Model:** Random Forest Classifier (200 trees)
- **Accuracy:** 72% (test set)
- **Training:** 70/30 split with 5-fold cross-validation
- **Features:** Automated encoding & imputation

---

## 🚨 **Alert System**

- 🟢 **Green:** AQI < 50 (Safe)
- 🟡 **Yellow:** AQI 50-100 (Moderate)
- 🔴 **Red:** AQI 100-200 (Poor)
- 🔴 **Very Poor:** AQI 200-300
- 🟣 **Severe:** AQI > 300

---

## 📥 **Exporting Data**

All pages support:
- 📊 **CSV Download** - Filtered or full dataset
- 📄 **Text Reports** - Formatted summaries
- 📉 **Chart Export** - PNG/JPG with metadata

**Download buttons available in:**
- Sidebar (global filtered data)
- Each page (specific reports)
- Reports page (comprehensive exports)

---

## ⚙️ **System Requirements**

### **Minimum (Mobile/Tablet Viewing):**
- Internet browser (Chrome, Safari, Firefox, Edge)
- 50MB RAM
- 10MB storage

### **Recommended (Running Dashboard):**
- Python 3.10+
- 4GB RAM
- 500MB disk space
- Windows/Mac/Linux

---

## 🐛 **Troubleshooting**

### **"Port 8503 already in use"**
```bash
streamlit run streamlit_app.py --server.port 8504
```

### **"Module not found" error**
```bash
pip install -r requirements.txt
```

### **Dashboard won't load on phone**
1. Check laptop IP: `ipconfig` (Windows)
2. Ensure both on same WiFi
3. Try: `http://<ip>:8503` (not localhost)
4. Check firewall settings

### **Charts not displaying on mobile**
- Mobile browsers might cache old version
- Clear browser cache: Settings → Clear browsing data
- Or use **incognito/private** mode

---

## 📞 **Support**

For issues:
1. Check terminal for error messages
2. Verify all dependencies installed: `pip list`
3. Restart dashboard: `Ctrl+C` then run again
4. Check internet connection

---

## 📈 **Performance Tips**

### **Mobile:**
- Use WiFi for faster loading
- Close other apps for smoother scrolling
- Avoid opening too many charts simultaneously

### **Tablet:**
- Tap hamburger (☰) to collapse sidebar for more space
- Landscape orientation gives wider views

### **Desktop:**
- Use modern browser (Chrome/Edge recommended)
- Maximize browser window for best experience
- Dual monitors: Dashboard on one, reference on other

---

## 🎓 **Learning Resources**

Inside dashboard:
- 📖 **Explainability Page:** Learn how predictions work
- 📊 **Model Performance Page:** Understand model reliability
- 💡 **Executive Summary:** High-level insights
- 🚨 **Early Warning:** Real-time monitoring

---

## ✅ **Checklist**

Before using:
- [ ] Python 3.10+ installed
- [ ] `pip install -r requirements.txt` run
- [ ] CSV file present: `air_pollution_50000_rows.csv`
- [ ] Model file present: `model.pkl`
- [ ] Port 8503 available (or change)
- [ ] Tested on your device

---

## 🌟 **Features Highlight**

✨ **14 interactive pages**  
✨ **72% accurate predictions**  
✨ **Real-time alerts**  
✨ **SHAP explainability**  
✨ **Policy what-if scenarios**  
✨ **Mobile-first responsive design**  
✨ **Cost-benefit analysis**  
✨ **Data quality tracking**  
✨ **Downloadable reports**  
✨ **Dark-mode ready**  

---

**Happy analyzing! 🎉**

*Last Updated: December 19, 2025*
*Version: 0.1.0*
