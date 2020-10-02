import React from 'react';
import Button from 'react-bootstrap/Button';

const RegistrationForm = () => {
    return (
        <div className='registration-form'>
            <h2>Register</h2>
            <p className='registration-info'>
                Register to upload videos of your favourite instructor as well as yourself to get constructive
                feedback on your poses during yoga.
            </p>
            <form action='/register_request'>
                <label for='email_address'>Email address</label><br/>
                <input type='text' id='email_address' name='email_address'/><br/>
                <label style={{paddingTop:'20%'}} for='password'>Password</label><br/>
                <input type='text' id='password' name='password'/><br/>
                <label for='confirm_password'>Confirm password</label><br/>
                <input type='text' id='confirm_password' name='confirm_password'/><br/>
                <Button type='submit'>Submit</Button>
            </form>
        </div>
    );
};

export default RegistrationForm;