import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import * as serviceWorker from './serviceWorker';
import WebcamStreamCapture from './WebcamStreamCapture';

ReactDOM.render(<WebcamStreamCapture />, document.getElementById("root"));

serviceWorker.unregister();
