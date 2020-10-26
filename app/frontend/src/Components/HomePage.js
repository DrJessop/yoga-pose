import React from 'react';
import TrackVisibility from 'react-on-screen';

import HomeInstructions from './instructions/home-instructions';
import yoga_output from '../videos/yoga_output.mp4';

const HomePage = () => {
    return (
        <>
            <video id='yoga-vid' className='videos' muted autoPlay loop>
                <source src={yoga_output} type='video/mp4' />
            </video>
            <center><p className='vid-text'>Find your flow</p></center>
            <TrackVisibility className='instructions'>
                {({ isVisible }) => isVisible && <div className="animate__animated animate__fadeInUp"><HomeInstructions/></div>}
            </TrackVisibility>
        </>
    )
}

export default HomePage;