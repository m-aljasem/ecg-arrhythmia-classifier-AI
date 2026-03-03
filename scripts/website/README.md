# ECG Classification Website

A modern, responsive static website for the ECG Classification project.

## 🌐 Pages

### 1. **index.html** - Home Page
- Hero section with call-to-action buttons
- Project statistics (89% accuracy, 21K+ records, etc.)
- Key features showcase
- Model architectures overview
- Technology stack
- Diagnostic classes explained
- Footer with links and disclaimer

### 2. **team.html** - Team Page
- Team member profiles with avatars
- Roles and bios
- Social links placeholders
- Mission and values section
- Contributions statistics
- Acknowledgments
- Join community CTA

### 3. **app.html** - Live Application Page
- Empty page ready for Streamlit Cloud iframe embed
- Detailed deployment instructions
- Example embed code
- Placeholder design with helpful tips

## 🎨 Design Features

- **Color Scheme:** Red (#dc143c) and White theme
- **Responsive Design:** Mobile-first approach with breakpoints
- **Modern UI:** Clean cards, smooth animations, shadows
- **Navigation:** Sticky top navbar with mobile menu toggle
- **Typography:** Segoe UI font family
- **Icons:** Emoji-based icons for modern feel

## 📁 File Structure

```
website/
├── index.html          # Home page
├── team.html           # About team page  
├── app.html            # Streamlit app embed page
├── css/
│   └── style.css      # Main stylesheet
├── js/
│   └── main.js        # JavaScript for interactivity
└── images/            # Image assets (empty, ready for use)
```

## 🚀 Usage

### Local Testing

Simply open any HTML file in your browser:
```bash
cd website
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows
```

Or use a local server:
```bash
# Python 3
python -m http.server 8000

# Then visit: http://localhost:8000
```

### Deploying to Web

#### GitHub Pages
1. Push the `website/` folder to your GitHub repository
2. Go to Settings → Pages
3. Select branch and `/website` folder
4. Your site will be at: `https://username.github.io/repo-name/`

#### Netlify
1. Drag and drop the `website/` folder to [Netlify Drop](https://app.netlify.com/drop)
2. Or connect your GitHub repo and set build folder to `website/`

#### Vercel
```bash
cd website
vercel --prod
```

## 🔗 Embedding Streamlit App

Once you deploy your Streamlit app to Streamlit Cloud:

1. Get your app URL (e.g., `https://your-app.streamlit.app`)
2. Open `app.html`
3. Find the commented-out iframe section at the bottom
4. Uncomment it and replace `YOUR-APP-URL` with your actual URL
5. Delete or hide the placeholder div

Example:
```html
<iframe 
    class="app-frame" 
    src="https://your-app.streamlit.app?embedded=true"
    title="ECG Classification Application"
></iframe>
```

**Important:** Enable iframe embedding in your Streamlit Cloud settings!

## ✏️ Customization

### Update Team Members
Edit `team.html` and modify the team cards:
- Change avatar letters in `<div class="team-avatar">A</div>`
- Update names, roles, and bios
- Add real social media links

### Change Colors
Edit `css/style.css` and modify the CSS variables:
```css
:root {
    --primary-red: #dc143c;    /* Main red color */
    --dark-red: #b71c1c;       /* Darker shade */
    --light-red: #ef5350;      /* Lighter shade */
    /* ... */
}
```

### Add Images/Logo
1. Place images in `website/images/` folder
2. Update logo in navigation:
```html
<a href="index.html" class="logo">
    <img src="images/logo.png" alt="Logo" style="width: 35px; height: 35px;">
    ECG Classifier
</a>
```

### Update Links
- GitHub repo links: Search for `https://github.com` and replace
- Update footer information
- Add real social media links in team page

## 📱 Responsive Breakpoints

- **Desktop:** 1200px+
- **Tablet:** 768px - 1199px  
- **Mobile:** < 768px

The design adapts automatically with:
- Collapsible mobile menu
- Stacked cards on small screens
- Adjusted font sizes
- Full-width buttons on mobile

## 🎯 Features

- ✅ Fully responsive design
- ✅ Smooth scroll animations
- ✅ Mobile-friendly navigation
- ✅ Fast loading (no heavy dependencies)
- ✅ SEO-friendly meta tags
- ✅ Accessible HTML structure
- ✅ Professional red/white theme
- ✅ Ready for Streamlit embed

## 📄 Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## 🔧 No Build Process Required

This is a static website - no Node.js, npm, or build tools needed! Just HTML, CSS, and vanilla JavaScript.

## 📝 License

Part of the ECG Classification Project - MIT License

## 🤝 Contributing

Feel free to customize and enhance the website for your needs!

---

**Need Help?** Check the comments in each HTML file for guidance on customization.
