const { app, BrowserWindow, session, ipcMain, desktopCapturer } = require('electron');
const path = require('path');

let win;

function createWindow() {
  win = new BrowserWindow({
    fullscreen: true,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  win.loadFile('index.html');

  win.setContentProtection(true);
  win.setIgnoreMouseEvents(true);
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