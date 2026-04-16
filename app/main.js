const { app, BrowserWindow, session, ipcMain, desktopCapturer } = require('electron');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile('index.html');
}

app.whenReady().then(() => {
  session.defaultSession.setDisplayMediaRequestHandler((request, callback) => {
    callback({ video: true, audio: false });
  });

  ipcMain.handle('get-sources', async () => {
    return await desktopCapturer.getSources({ types: ['screen'] });
  });

  createWindow();
});