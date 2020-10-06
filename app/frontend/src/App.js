import React, { Component } from 'react';
import wld from './images/wld.jpeg';
import { BrowserRouter as Router, Link, Route} from 'react-router-dom';
import HomePage from './Components/HomePage';
import Team from './Components/Team';
import UploadVid from './Components/upload/UploadVid';
import ProcessedVideos from './Components/ProcessedVideos';

import 'animate.css';
import './App.css';

const ids = ['team', 'videos', 'upload'];

function active_color(menu_id) {
    ids.forEach(id => document.getElementById(id).style.color = 'black');
    document.getElementById(menu_id).style.color = 'green';
}

function onLoadColor() {
    if (window.location.href.includes('meetTheTeam')) {
        document.getElementById('team').style.color = 'green';
    }
    else if (window.location.href.includes('processed_videos')) {
        document.getElementById('videos').style.color = 'green';
    }
    else if (window.location.href.includes('upload')) {
        document.getElementById('upload').style.color = 'green';
    }
}

class App extends Component {

    componentDidMount() {
        window.addEventListener('load', onLoadColor);
    }

    render() {
        return (
            <Router>
                <ul id='home-menu' className='home-menu'>
                    <li className='left'>
                        <Link id='logo' to='/' onClick={() => active_color('logo')}>
                            <img src={wld} className='wld'/>
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/meetTheTeam' id='team' className='right-text' onClick={() => active_color('team')}>
                            Meet the team
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/processed_videos' id='videos' className='right-text' onClick={() => active_color('videos')}>
                            Your Videos
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/upload' id='upload' className='right-text' onClick={() => active_color('upload')}>
                            Upload
                        </Link>
                    </li>
                </ul>
            <Route exact path='/' component={HomePage} />
            <Route exact path='/meetTheTeam' component={Team} />
            <Route exact path='/processed_videos' component={ProcessedVideos} />
            <Route exact path='/upload' component={UploadVid} />
            </Router>
        );
    }
}

export default App;