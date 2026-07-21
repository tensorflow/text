<?php

use App\Http\Controllers\LoginController;
use App\Http\Controllers\SaleController;
use App\Http\Controllers\LogoutController;
use App\Http\Controllers\UserController;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "web" middleware group. Make something great!
|
*/

Route::middleware('auth')->group(function () {
    // User Routes
    Route::get('/users', [UserController::class, 'index'])->name('userindex');
    Route::post('/add-new-user', [UserController::class, 'store']);
    Route::post('/update-user/{user}', [UserController::class, 'update']);
    Route::delete('/delete-user/{user}', [UserController::class, 'delete']);
    Route::match(['POST'], '/logout', LogoutController::class)->name('logout');

    //Sales Routes
    Route::resource('/', SaleController::class)->only(['index']);
    Route::post('/save-new-sale', [SaleController::class, 'save_new_sale']);
    Route::delete('/delete-sale', [SaleController::class, 'delete_sale']);
    Route::get('/trash', [SaleController::class, 'trash'])->name('trash');
    Route::get('/undo-sale/{sale}', [SaleController::class, 'undo_sale']);
    Route::post('/restore-sale', [SaleController::class, 'restore_sale']);
    Route::put('/live_update_sale', [SaleController::class, 'live_update_sale']);
    Route::get('/check_is_so_unique', [SaleController::class, 'check_is_so_unique']);
    Route::get('/add-new-sale-ui', [SaleController::class, 'addNewSaleUi']);
    Route::get('/last-update', [SaleController::class, 'last_update']);
});

Route::middleware('web')->group(function () {
    Route::middleware('guest')->group(function () {
        Route::match(['GET', 'POST'], '/login', LoginController::class)->name('login');
    });
});
