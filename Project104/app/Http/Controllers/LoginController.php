<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;

class LoginController extends Controller
{
    public function __invoke(Request $request)
    {
        $method = $request->method();
        if ($method == 'GET') {
            return view('login');
        }

        $inputs = $request->validate([
            'email' => ['required', 'email', 'max:255'],
            'password' => ['required', 'min:8', 'max:32'],
        ]);

        $loginSucceed = auth()->attempt($inputs);
        if (!$loginSucceed) {
            return redirect()->back()->withErrors([
                'error' => 'The email address or password is incorrect.',
            ]);
        }

        $request->session()->regenerate();

        return redirect('/')->with([
            'success' => 'success.',
        ]);

    }
}
