import React from 'react';

const HomeInstructions = () => {
    return (
        <div className='home-instructions'>
            <h2 className='home-instructions-header'><center>Only 3 steps</center></h2>
            <ul>
                <li className='home-instructions-steps'>
                    Register
                </li>
                <li className='home-instructions-steps'>
                    Drag and drop 
                </li>
                <li className='home-instructions-steps'>
                    Wait 
                </li>
            </ul>
        </div>
    );
}

export default HomeInstructions;