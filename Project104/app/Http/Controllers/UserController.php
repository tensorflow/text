<?php

namespace App\Http\Controllers;

use App\Mail\UserRegistered;
use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Mail;
use Illuminate\Support\Str;


class UserController extends Controller
{
    public function index()
    {
        if (auth()->user()->role == "superAdmin") {
            return view('users', [
                'users' => User::all(),
            ]);
        } else {
            return redirect('/');
        }
    }
    public function store(Request $request)
    {
        $request->validate([
            'name' => 'required|string|max:50',
            'email' => 'required|email|unique:users,email',
            'role' => 'required|string|max:10'
        ]);

        try {
            $user = new User();
            $user->name = $request->name;
            $user->email = $request->email;
            $user->role = $request->role;
            $password = Str::random(10);
            $user->password = Hash::make($password);
            $user->save();
            Mail::to($user->email)->send(new UserRegistered($password));
            return response()->json([
                'success' => 'success'
            ], 200);
        } catch (\Throwable $th) {
            return response()->json([
                'error' => $th
            ], 500);
        }
    }
    public function update(Request $request, User $user)
    {
        $request->validate([
            'name' => 'required|string|max:50',
            'email' => 'required|email|unique:users,email,' . $user->id,
            'role' => 'required|string|max:10'
        ]);
        try {
            $user->name = $request->name;
            $user->email = $request->email;
            $user->role = $request->role;
            $user->save();
            return response()->json([
                'success' => 'success'
            ], 200);
        } catch (\Throwable $th) {
            return response()->json([
                'error' => $th
            ], 500);
        }
    }
    public function delete(User $user)
    {
        try {
            $user->delete();
            return response()->json([
                'success' => 'success',
            ], 200);
        } catch (\Throwable $th) {
            return response()->json([
                'error' => 'error',
            ], 500);
        }
    }
}
