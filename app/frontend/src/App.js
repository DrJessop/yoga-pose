import React from 'react';
import 'animate.css';

import HomeMenu from './Components/menus/home-menu';
import './App.css';

function App() {

    return (
        <div id='main'>
            <HomeMenu logged_in={false}/>
        </div>
    );
}

export default App;
