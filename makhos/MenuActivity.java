package com.duckprog.makhos;

import java.io.IOException;
import java.io.InputStream;

import org.andengine.engine.camera.Camera;
import org.andengine.engine.options.EngineOptions;
import org.andengine.engine.options.ScreenOrientation;
import org.andengine.engine.options.resolutionpolicy.RatioResolutionPolicy;
//import org.andengine.entity.IEntityMatcher;
import org.andengine.entity.modifier.MoveModifier;
import org.andengine.entity.scene.IOnSceneTouchListener;
import org.andengine.entity.scene.Scene;
//import org.andengine.entity.scene.background.Background;
import org.andengine.entity.sprite.Sprite;
import org.andengine.entity.text.Text;
import org.andengine.entity.util.FPSLogger;
import org.andengine.input.touch.TouchEvent;
import org.andengine.opengl.font.Font;
import org.andengine.opengl.font.FontFactory;
import org.andengine.opengl.texture.ITexture;
import org.andengine.opengl.texture.bitmap.BitmapTexture;
import org.andengine.opengl.texture.region.ITextureRegion;
import org.andengine.opengl.texture.region.TextureRegionFactory;
import org.andengine.opengl.vbo.VertexBufferObjectManager;
import org.andengine.ui.activity.LayoutGameActivity;
import org.andengine.util.adt.io.in.IInputStreamOpener;



import com.apptracker.android.track.AppTracker;
//import com.yjvfxhmgkxadkxmw.AdController;
// 18/08/2014
//import com.isszzngerzmkgvzq.AdController;
//import com.ahddtejxswazwdqre.AdController;
//18/08/2014
import com.bheugyiqywscgscae.AdController;



import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Typeface;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.util.DisplayMetrics;
import android.view.Gravity;
import android.widget.RelativeLayout;


public class MenuActivity extends LayoutGameActivity implements IOnSceneTouchListener {

	private static int CAMERA_WIDTH = 840;
	private static int CAMERA_HEIGHT = 1200;
	
	private Font mFont;
	
	private ITextureRegion mBackgroundRegion, itr_newgame, itr_setup, itr_moreapp, itr_setuppanel;
	private ITextureRegion itr_piece1, itr_piece2, itr_piece3, itr_piece4,itr_statpanel , itr_statbut;
	private ITextureRegion itr_piece1dis, itr_piece2dis, itr_piece3dis, itr_piece4dis;
	private ITextureRegion itr_player1, itr_player2, itr_back, itr_disable, itr_enable, itr_disabledis,itr_enabledis;
	private ITextureRegion itr_text1; 
	private Sprite sAppStore;
	private AdController adwall;
	private ITextureRegion itr_appStore;
	private Sprite sNewGame , sSetup , sMoreApp , sSetupPanel , sPlayer1 , sPlayer2, sBack , sDisable, sDisableDis, sEnable , sEnableDis, sText1, sStatPanel, sStatBut;
	
	private Sprite[] sLevel = new Sprite[3];
	private Sprite[] sLevelDis = new Sprite[2];
	
	private Sprite[] sSelecePiece = new Sprite[4];
	private Sprite[] sSelecePieceDis = new Sprite[4];
	
	int level_select = 2;
	int piece_select = 0;
	
	private MediaPlayer mMenu;

	private boolean newGame = false;
	private boolean soundEnable = false;
	
	Scene mCurrentScene;

	private AdController ad;
	private String sectionid;
	private int pointTop;
	@Override
	protected void onDestroy(){
		ad.destroyAd();
		adwall.destroyAd();
		super.onDestroy();
	}
	
	@Override
	protected void onCreate(Bundle pSavedInstanceState) {
		super.onCreate(pSavedInstanceState);
		System.out.println("-------------- App Store Process");
		if(pSavedInstanceState == null) {
            // Initialize Leadbolt SDK with your api key
			System.out.println("-------------- App Store Process 2");
			AppTracker.startSession(getApplicationContext(),"lfli7sPXV7MIL1u9azv8dzjqc7ENU6ty");
        }
        // cache Leadbolt Ad without showing it
        AppTracker.loadModuleToCache(getApplicationContext(),"inapp");
		DisplayMetrics dm = new DisplayMetrics();
		  getWindowManager().getDefaultDisplay().getMetrics(dm);
		  int screenWidth = dm.widthPixels;
		  pointTop = dm.widthPixels/2-420;
		  /*sectionid = "383773578";
		  if(screenWidth >= 720) {
		  sectionid = "308799023";
		  }
		  else if(screenWidth >= 640) {
		  sectionid = "245361661";
		  }
		  else if(screenWidth >= 468) {
		  sectionid = "909494006";
		  }*/
		  sectionid = "134531518";

		  ad = new AdController(this, sectionid);
		  /*if (pointTop>0){
			  ad.setAdditionalDockingMargin(pointTop);
		  }
		  else{
			  ad.setAdditionalDockingMargin(0);
		  }*/
		  ad.setAdditionalDockingMargin(0);
		  adwall = new AdController(this, "441825600");
		  ad.loadAd();
		  
	}

	@Override
	public void onResume()
	{
		super.onResume();
		ad.destroyAd();
		ad = new AdController(this, sectionid);
		/*if (pointTop>0){
			ad.setAdditionalDockingMargin(pointTop);
		}
		else{
			ad.setAdditionalDockingMargin(0);
		}*/
		ad.setAdditionalDockingMargin(0);
		ad.loadAd();//*/
		
		adwall.destroyAd();
		adwall = new AdController(this, "441825600");

	}
	
	public EngineOptions onCreateEngineOptions() {
		// TODO Auto-generated method stub
		final Camera camera = new Camera(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);

		return new EngineOptions(false, ScreenOrientation.PORTRAIT_FIXED,
				new RatioResolutionPolicy(CAMERA_WIDTH, CAMERA_HEIGHT), camera);
	}

	public void onCreateResources(
			OnCreateResourcesCallback pOnCreateResourcesCallback)
			throws Exception {
		// TODO Auto-generated method stub
		try {
	
			ITexture backgroundTexture = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/firstpageNew.jpg");
						}
	
					});
			ITexture iNewGame = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/newgame.png");
						}
	
					});
			ITexture iSetup = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/setup.png");
						}
	
					});
			
			ITexture iMoreApp = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/moreapp.png");
						}
	
					});
			
			ITexture iSetupPanel = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/setuppanel.png");
						}
	
					});
			
			ITexture iPiece1 = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece1.png");
						}
	
					});
			
			ITexture iPiece1dis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece1dis.png");
						}
	
					});
			
			ITexture iPiece2 = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece2.png");
						}
	
					});
			
			ITexture iPiece2dis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece2dis.png");
						}
	
					});
			
			ITexture iPiece3 = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece3.png");
						}
	
					});
			
			ITexture iPiece3dis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece3dis.png");
						}
	
					});
			
			ITexture iPiece4 = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece4.png");
						}
	
					});
			
			ITexture iPiece4dis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/piece4dis.png");
						}
	
					});
			
			ITexture iPlayer1 = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/player1n.png");
						}
	
					});
			
			
			ITexture iPlayer2 = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/player2n.png");
						}
	
					});
			
			ITexture iBack = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/backtomenu.png");
						}
	
					});
			
			ITexture iEnable = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/enable.png");
						}
	
					});
			
			ITexture iEnableDis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/enabledis.png");
						}
	
					});
			
			ITexture iDisable = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/disable.png");
						}
	
					});
			
			ITexture iDisableDis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/disabledis.png");
						}
	
					});
			ITexture iText1 = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/text1.png");
						}
	
					});
			
			ITexture iStatBut = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/stat.png");
						}
	
					});
			
			ITexture iStatPanel = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/statpanel.png");
						}
	
					});
			
			ITexture iAppStore = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/appstore.png");
						}
	
					});
			
			backgroundTexture.load();
			iNewGame.load();
			iSetup.load();
			iMoreApp.load();
			iAppStore.load();
			iSetupPanel.load();
			iPiece1.load();
			iPiece1dis.load();
			iPiece2.load();
			iPiece2dis.load();
			iPiece3.load();
			iPiece3dis.load();
			iPiece4.load();
			iPiece4dis.load();
			iPlayer1.load();
			iPlayer2.load();
			iBack.load();
			iEnable.load();
			iEnableDis.load();
			iDisable.load();
			iDisableDis.load();
			iText1.load();
			iStatBut.load();
			iStatPanel.load();
			
			this.itr_newgame = TextureRegionFactory.extractFromTexture(iNewGame);
			this.itr_setup = TextureRegionFactory.extractFromTexture(iSetup);
			this.itr_moreapp = TextureRegionFactory.extractFromTexture(iMoreApp);
			this.itr_appStore = TextureRegionFactory.extractFromTexture(iAppStore);
			this.itr_setuppanel = TextureRegionFactory.extractFromTexture(iSetupPanel);
			this.itr_piece1 = TextureRegionFactory.extractFromTexture(iPiece1);
			this.itr_piece1dis = TextureRegionFactory.extractFromTexture(iPiece1dis);
			this.itr_piece2 = TextureRegionFactory.extractFromTexture(iPiece2);
			this.itr_piece2dis = TextureRegionFactory.extractFromTexture(iPiece2dis);
			this.itr_piece3 = TextureRegionFactory.extractFromTexture(iPiece3);
			this.itr_piece3dis = TextureRegionFactory.extractFromTexture(iPiece3dis);
			this.itr_piece4 = TextureRegionFactory.extractFromTexture(iPiece4);
			this.itr_piece4dis = TextureRegionFactory.extractFromTexture(iPiece4dis);
			this.itr_player1 = TextureRegionFactory.extractFromTexture(iPlayer1);
			this.itr_player2 = TextureRegionFactory.extractFromTexture(iPlayer2);
			this.itr_back = TextureRegionFactory.extractFromTexture(iBack);
			
			this.itr_enable = TextureRegionFactory.extractFromTexture(iEnable);
			this.itr_enabledis = TextureRegionFactory.extractFromTexture(iEnableDis);
			this.itr_disable = TextureRegionFactory.extractFromTexture(iDisable);
			this.itr_disabledis = TextureRegionFactory.extractFromTexture(iDisableDis);
			this.itr_text1 = TextureRegionFactory.extractFromTexture(iText1);
			
			this.itr_statpanel = TextureRegionFactory.extractFromTexture(iStatPanel);
			this.itr_statbut = TextureRegionFactory.extractFromTexture(iStatBut);
			
			this.mFont = FontFactory.create(this.getFontManager(), this.getTextureManager(), 256, 256, Typeface.create(Typeface.DEFAULT, Typeface.BOLD), 32 );
			this.mFont.load();
			
			this.mBackgroundRegion = TextureRegionFactory
					.extractFromTexture(backgroundTexture);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		mMenu = MediaPlayer.create(getBaseContext(), R.raw.menu);
		mMenu.setLooping(false);
		pOnCreateResourcesCallback.onCreateResourcesFinished();
		//*/
	}
	
	public void onCreateScene(OnCreateSceneCallback pOnCreateSceneCallback)
			throws Exception {
		// TODO Auto-generated method stub	
		
		this.mEngine.registerUpdateHandler(new FPSLogger());
		mCurrentScene = new Scene();
		Sprite bg = new Sprite(0, 0, this.mBackgroundRegion,
				getVertexBufferObjectManager());
		mCurrentScene.attachChild(bg);
		initButton();
		mCurrentScene.setOnSceneTouchListener(this);
		pOnCreateSceneCallback.onCreateSceneFinished(mCurrentScene);
	}

	public void onPopulateScene(Scene pScene,
			OnPopulateSceneCallback pOnPopulateSceneCallback) throws Exception {
		// TODO Auto-generated method stub
		pOnPopulateSceneCallback.onPopulateSceneFinished();
	}

	@Override
	protected int getLayoutID() {
		// TODO Auto-generated method stub
		return R.layout.activity_menu;
	}

	@Override
	protected int getRenderSurfaceViewID() {
		// TODO Auto-generated method stub
		return R.id.andengineID;
	}

	private void initButton(){
		
		sNewGame = new Sprite( 195 , 300 , this.itr_newgame , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					newGame = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		
		sStatBut = new Sprite( 195 , 450 , this.itr_statbut , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					if (!sStatPanel.isVisible()){
						sAppStore.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 1020 ,1170));
						sMoreApp.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 750 ,1020));
						sSetup.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 750 ,870));
						sStatPanel.setVisible(true);
						sSetupPanel.setVisible(false);
					}
					else{
						sAppStore.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 1020 ,900));
						sMoreApp.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 1020 ,750));
						sSetup.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 750 ,600));
						sStatPanel.setVisible(false);
						sSetupPanel.setVisible(false);
					}
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		
		SharedPreferences app_preferences = 
		         PreferenceManager.getDefaultSharedPreferences(this);
		        
		int l1win = app_preferences.getInt("l1win", 0);
		int l1draw = app_preferences.getInt("l1draw", 0);
		int l1lose = app_preferences.getInt("l1lose", 0);
		
		int l2win = app_preferences.getInt("l2win", 0);
		int l2draw = app_preferences.getInt("l2draw", 0);
		int l2lose = app_preferences.getInt("l2lose", 0);
		
		int l3win = app_preferences.getInt("l3win", 0);
		int l3draw = app_preferences.getInt("l3draw", 0);
		int l3lose = app_preferences.getInt("l3lose", 0);
		
		VertexBufferObjectManager vertexBufferObjectManager = this.getVertexBufferObjectManager();
		Text centerText1 = new Text(420, 80, this.mFont, l1win+"  /  "+l1draw+"  /  "+l1lose, vertexBufferObjectManager);		
		Text centerText2 = new Text(420, 150, this.mFont, l2win+"  /  "+l2draw+"  /  "+l2lose, vertexBufferObjectManager);
		Text centerText3 = new Text(420, 220, this.mFont, l3win+"  /  "+l3draw+"  /  "+l3lose, vertexBufferObjectManager);
		
		sStatPanel = new Sprite( 65 , 575 , this.itr_statpanel , getVertexBufferObjectManager());
		
		sStatPanel.attachChild(centerText1);
		sStatPanel.attachChild(centerText2);
		sStatPanel.attachChild(centerText3);
		
		sStatPanel.setVisible(false);
		mCurrentScene.registerTouchArea(sStatPanel);
		mCurrentScene.attachChild(sStatPanel);
		
		
		mCurrentScene.registerTouchArea(sStatBut);
		mCurrentScene.attachChild(sStatBut);
		
		sSetup = new Sprite( 195 , 600 , this.itr_setup , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					if (!sSetupPanel.isVisible()){
					//if (sMoreApp.getY() != 1020){
						if (sSetup.getY()!=600){
							sSetup.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 750 ,600));
						}
						sAppStore.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 1020 ,1170));
						sMoreApp.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 750 ,1020));
						sSetupPanel.setVisible(true);
						sStatPanel.setVisible(false);
					}
					else{
						sAppStore.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 1020 ,900));
						sMoreApp.registerEntityModifier(new MoveModifier((float) 0.2, 195, 195 , 1020 ,750));
						sSetupPanel.setVisible(false);
						sStatPanel.setVisible(false);
					}
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		//AdController adwall;
		
		  /*if (pointTop>0){
			  ad.setAdditionalDockingMargin(pointTop);
		  }
		  else{
			  ad.setAdditionalDockingMargin(0);
		  }*/

		  //ad.loadAd();
		sMoreApp = new Sprite( 195 , 750 , this.itr_moreapp , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					String APP_MARKET_URL = "market://search?q=pub:sengpedman";

					Intent intent = new Intent(Intent.ACTION_VIEW,
					Uri.parse(APP_MARKET_URL));
					intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
					startActivity(intent);
					//adwall.loadAd();

					//441825600
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		
		sAppStore = new Sprite( 190 , 900 , itr_appStore , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					
					//System.out.println("----- App Store Process");
					adwall.loadAd();
					
					
					//finish();
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};//*/
		
		sSetupPanel = new Sprite( 65 , 725 , this.itr_setuppanel , getVertexBufferObjectManager());
		
		sLevel[0] = new Sprite( 30 , 180 , this.itr_piece2 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("level_select", 2);
					        level_select = 2;
					        editor.commit();
					sLevel[1].setVisible(false);
					sLevel[2].setVisible(false);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		
		sLevelDis[0] = new Sprite( 160 , 180 , this.itr_piece2dis , getVertexBufferObjectManager());
		sLevel[1] = new Sprite( 160 , 180 , this.itr_piece2 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("level_select", 4);
					        level_select = 4;
					        editor.commit();
					this.setVisible(true);
					sLevel[2].setVisible(false);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		sLevel[1].setVisible(false);
		
		sLevelDis[1] = new Sprite( 290 , 180 , this.itr_piece2dis , getVertexBufferObjectManager());
		sLevel[2] = new Sprite( 290 , 180 , this.itr_piece2 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("level_select", 6);
					        level_select = 6;
					        editor.commit();
					sLevel[1].setVisible(true);
					this.setVisible(true);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		sLevel[2].setVisible(false);
		
		sSetupPanel.attachChild(sLevel[0]);
		mCurrentScene.registerTouchArea(sLevel[0]);
		sSetupPanel.attachChild(sLevelDis[0]);
		sSetupPanel.attachChild(sLevel[1]);
		mCurrentScene.registerTouchArea(sLevel[1]);
		sSetupPanel.attachChild(sLevelDis[1]);
		sSetupPanel.attachChild(sLevel[2]);
		mCurrentScene.registerTouchArea(sLevel[2]);
		
		/*SharedPreferences app_preferences = 
		         PreferenceManager.getDefaultSharedPreferences(this);*/
		        
		level_select = app_preferences.getInt("level_select", 4);
		
		if (level_select == 4){
			sLevel[1].setVisible(true);
			sLevel[2].setVisible(false);
		}
		else if (level_select == 6){
			sLevel[1].setVisible(true);
			sLevel[2].setVisible(true);
		}
		
		sSelecePieceDis[0] = new Sprite( 440 , 70 , this.itr_piece2dis , getVertexBufferObjectManager());
		sSelecePiece[0] = new Sprite( 440 , 70 , this.itr_piece2 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("piece_select", 0);
					        piece_select = 0;
					        editor.commit();
					sSelecePiece[0].setVisible(true);
					sSelecePiece[1].setVisible(false);
					sSelecePiece[2].setVisible(false);
					sSelecePiece[3].setVisible(false);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		sSetupPanel.attachChild(sSelecePieceDis[0]);
		sSetupPanel.attachChild(sSelecePiece[0]);
		mCurrentScene.registerTouchArea(sSelecePiece[0]);
		
		sSelecePieceDis[1] = new Sprite( 570 , 70 , this.itr_piece1dis , getVertexBufferObjectManager());
		sSelecePiece[1] = new Sprite( 570 , 70 , this.itr_piece1 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("piece_select", 1);
					        piece_select = 1;
					        editor.commit();
					sSelecePiece[0].setVisible(false);
					sSelecePiece[1].setVisible(true);
					sSelecePiece[2].setVisible(false);
					sSelecePiece[3].setVisible(false);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		sSelecePiece[1].setVisible(false);
		sSetupPanel.attachChild(sSelecePieceDis[1]);
		sSetupPanel.attachChild(sSelecePiece[1]);
		mCurrentScene.registerTouchArea(sSelecePiece[1]);
		
		sSelecePieceDis[2] = new Sprite( 440 , 180 , this.itr_piece3dis , getVertexBufferObjectManager());
		sSelecePiece[2] = new Sprite( 440 , 180 , this.itr_piece3 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("piece_select", 2);
					        piece_select = 2;
					        editor.commit();
					sSelecePiece[0].setVisible(false);
					sSelecePiece[1].setVisible(false);
					sSelecePiece[2].setVisible(true);
					sSelecePiece[3].setVisible(false);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		sSelecePiece[2].setVisible(false);
		sSetupPanel.attachChild(sSelecePieceDis[2]);
		sSetupPanel.attachChild(sSelecePiece[2]);
		mCurrentScene.registerTouchArea(sSelecePiece[2]);
		
		sSelecePieceDis[3] = new Sprite( 570 , 180 , this.itr_piece4dis , getVertexBufferObjectManager());
		sSelecePiece[3] = new Sprite( 570 , 180 , this.itr_piece4 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("piece_select", 3);
					        piece_select = 3;
					        editor.commit();
					sSelecePiece[0].setVisible(false);
					sSelecePiece[1].setVisible(false);
					sSelecePiece[2].setVisible(false);
					sSelecePiece[3].setVisible(true);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		sSelecePiece[3].setVisible(false);
		sSetupPanel.attachChild(sSelecePieceDis[3]);
		sSetupPanel.attachChild(sSelecePiece[3]);
		mCurrentScene.registerTouchArea(sSelecePiece[3]);
		
		piece_select = app_preferences.getInt("piece_select", 0);
		
		if (piece_select == 1){
			sSelecePiece[0].setVisible(false);
			sSelecePiece[1].setVisible(true);
			sSelecePiece[2].setVisible(false);
			sSelecePiece[3].setVisible(false);
		}else if (piece_select == 2){
			sSelecePiece[0].setVisible(false);
			sSelecePiece[1].setVisible(false);
			sSelecePiece[2].setVisible(true);
			sSelecePiece[3].setVisible(false);
		}else if (piece_select == 3){
			sSelecePiece[0].setVisible(false);
			sSelecePiece[1].setVisible(false);
			sSelecePiece[2].setVisible(false);
			sSelecePiece[3].setVisible(true);
		}
		
		sEnableDis = new Sprite( 40 , 70 , this.itr_enabledis , getVertexBufferObjectManager());
		sEnable = new Sprite( 40 , 70 , this.itr_enable , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putBoolean("sound_enable", true);
					        soundEnable = true;
					        editor.commit();
					this.setVisible(true);
					sDisable.setVisible(false);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		
		sDisableDis = new Sprite( 220 , 70 , this.itr_disabledis , getVertexBufferObjectManager());
		sDisable = new Sprite( 220 , 70 , this.itr_disable , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					         PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putBoolean("sound_enable", false);
					        soundEnable = false;
					        editor.commit();
					this.setVisible(true);
					sEnable.setVisible(false);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		sDisable.setVisible(false);
		
		sSetupPanel.attachChild(sEnableDis);
		sSetupPanel.attachChild(sDisableDis);
		
		sSetupPanel.attachChild(sEnable);
		mCurrentScene.registerTouchArea(sEnable);
		sSetupPanel.attachChild(sDisable);
		mCurrentScene.registerTouchArea(sDisable);
		
		soundEnable= app_preferences.getBoolean("sound_enable", true);
		
		if (soundEnable){
			sEnable.setVisible(true);
			sDisable.setVisible(false);
		}
		else{
			sEnable.setVisible(false);
			sDisable.setVisible(true);
		}
		
		mCurrentScene.attachChild(sNewGame);
		mCurrentScene.registerTouchArea(sNewGame);
		mCurrentScene.attachChild(sSetup);
		mCurrentScene.registerTouchArea(sSetup);
		mCurrentScene.attachChild(sMoreApp);
		mCurrentScene.registerTouchArea(sMoreApp);
		mCurrentScene.attachChild(sAppStore);
		mCurrentScene.registerTouchArea(sAppStore);
		
		sSetupPanel.setVisible(false);
		mCurrentScene.attachChild(sSetupPanel);
	}

	private void initPlayerButton(){
		mCurrentScene.clearTouchAreas();
		sStatPanel.setVisible(false);
		sStatBut.setVisible(false);
		sSetupPanel.setVisible(false);
		sSetup.setVisible(false);
		sMoreApp.setVisible(false);
		sAppStore.setVisible(false);
		sNewGame.setVisible(false);	

		sPlayer1 = new Sprite( 195 , 380 , itr_player1 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					        PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("level_select", level_select);
		
							editor.putInt("piece_select", piece_select);
							
							editor.putBoolean("player2", false);
							
							editor.putBoolean("sound_enable", soundEnable);
							
							editor.commit();
						Intent i = new Intent(getApplicationContext() , MainActivity.class);
						startActivity(i);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};

		sPlayer2 = new Sprite( 195 , 580 , itr_player2 , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					SharedPreferences app_preferences = 
					        PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

					        SharedPreferences.Editor editor = app_preferences.edit();
					        editor.putInt("level_select", level_select);
		
							editor.putInt("piece_select", piece_select);
							
							editor.putBoolean("player2", true);
							
							editor.putBoolean("sound_enable", soundEnable);
							
							editor.commit();
						Intent i = new Intent(getApplicationContext() , MainActivity.class);
						startActivity(i);
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		
		sText1 = new Sprite(25 , 680 , itr_text1 , getVertexBufferObjectManager());
		
		sBack = new Sprite( 195 , 880 , itr_back , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {			
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					if(soundEnable){
						mMenu.start();
					}
					sPlayer1.setVisible(false);
					sPlayer2.setVisible(false);
					sText1.setVisible(false);
					this.setVisible(false);
					mCurrentScene.clearTouchAreas();
					initButton();
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};

		mCurrentScene.attachChild(sPlayer1);
		mCurrentScene.attachChild(sPlayer2);
		mCurrentScene.attachChild(sText1);
		mCurrentScene.attachChild(sBack);
		mCurrentScene.registerTouchArea(sPlayer1);
		mCurrentScene.registerTouchArea(sPlayer2);
		mCurrentScene.registerTouchArea(sBack);
	}
	
	@Override
	public boolean onSceneTouchEvent(Scene pScene, TouchEvent pSceneTouchEvent) {
		// TODO Auto-generated method stub
		if (newGame == true){
			initPlayerButton();
			newGame = false;
		}
		return false;
	}

	void detach(final Sprite spr){
		mEngine.runOnUpdateThread(new Runnable() {
			@Override
			public void run() {
				//mCurrentScene.detachChildren();
				spr.detachSelf();
			}
		});
	}
}
