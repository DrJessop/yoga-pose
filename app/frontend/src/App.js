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

class App extends Component {

    active_color(menu_id) {
        ids.forEach(id => document.getElementById(id).style.color = 'black');
        document.getElementById(menu_id).style.color = 'green';
    }
    
    onLoadColor() {
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

    updateCardsFromTitle(filePath) {
        var fileSplit = filePath.split('/');
        var fileName = fileSplit[fileSplit.length - 1];
        this.updateCards(fileName, '', 'Oct 12, 2020', filePath);
    }
    
    onLoadCards() {
        fetch('http://127.0.0.1:5000/get_overlaps')
          .then(response => response.json())
          .then(response => JSON.parse(response.files).forEach(file => this.updateCardsFromTitle(file)));
    }
    
    onLoad() {
        this.onLoadColor();
        this.onLoadCards();
    }

    componentDidMount() {
        window.addEventListener('load', this.onLoad());
    }

    constructor() {
        super();
        this.state = {title: [], link: [], date: [], path: []};
    }

    updateCards(title, link, date, path) {
        var newtitle = this.state.title.concat(title);
        var newlink  = this.state.link.concat(link);
        var newdate  = this.state.date.concat(date);
        var newpath  = this.state.path.concat(path)
        this.setState({title: newtitle, link: newlink, date: newdate, path: newpath});
        console.log(this.state);
    }

    render() {
        return (
            <Router>
                <ul id='home-menu' className='home-menu'>
                    <li className='left'>
                        <Link id='logo' to='/' onClick={() => this.active_color('logo')}>
                            <img src={wld} className='wld'/>
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/meetTheTeam' id='team' className='right-text' onClick={() => this.active_color('team')}>
                            Meet the team
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/processed_videos' id='videos' className='right-text' onClick={() => this.active_color('videos')} >
                            Your Videos
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/upload' id='upload' className='right-text' onClick={() => this.active_color('upload')}>
                            Upload
                        </Link>
                    </li>
                </ul>
            <Route exact path='/' component={HomePage} />
            <Route exact path='/meetTheTeam' component={Team} />
            <Route exact path='/processed_videos' component={() => <ProcessedVideos title={this.state.title} 
                                                                                    link={this.state.link}
                                                                                    date={this.state.date}
                                                                                    path={this.state.path}/>} />
            <Route exact path='/upload' component={() => <UploadVid updateCards={(title, link, date, path) => 
                                                                                    this.updateCards(title, link, date, path)} />}/>
            </Router>
        );
    }
}

export default App;