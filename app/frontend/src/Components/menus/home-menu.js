import React from 'react';
import wld from '../../images/wld.jpeg';
import { BrowserRouter as Router, Link, Route} from 'react-router-dom';
/* import RegistrationForm from '../forms/registration'; */
import HomePage from '../HomePage';
import AboutPage from '../AboutPage';
import UploadVid from '../upload/UploadVid';

const HomeMenu = (props) => {
    var login_message;
    if (props.logged_in) {
        login_message = 'Logout';
    }
    else {
        login_message = 'Login';
    }
    return (
        <Router>
            <ul id='home-menu' className='home-menu'>
                <li className='left'>
                    <Link to='/'>
                        <img src={wld} className='wld'/>
                    </Link>
                </li>
                <li className='right'>
                    <Link to='/about' className='right-text'>
                        About
                    </Link>
                </li>
                <li className='right'>
                    <Link to='/login' className='right-text'>
                        {login_message}
                    </Link>
                </li>
                <li className='right'>
                    <Link to='/upload' className='right-text'>
                        Upload
                    </Link>
                </li>
            </ul>
        <Route exact path='/' component={HomePage} />
        {/* <Route exact path='/register' component={RegistrationForm} /> */}
        <Route exact path='/about' component={AboutPage} />
        <Route exact path='/upload' component={UploadVid} />
        </Router>
    );
}

export default HomeMenu;