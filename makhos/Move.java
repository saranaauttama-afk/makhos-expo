package com.duckprog.makhos;

import java.util.Vector;

//import com.duckprog.makhos.Engine;
import com.duckprog.makhos.MainActivity;

public class Move {
	static Vector<int[]> endList=new Vector<int[]>();
	//static Vector<int[]> initList=new Vector<int[]>();
	final static int legalMove=1;
	final static int illegalMove=2;
	final static int incompleteMove=3;
	static int ch1=0,ch2=0;
	
	static int applyMove(int[][] board,int startI,int startJ,int endI,int endJ,boolean human)
	{
		if (human){
			if (board[startI][startJ] == MainActivity.blackKing || board[startI][startJ] == MainActivity.whiteKing)
				MainActivity.countDraw++;
			else
				MainActivity.countDraw = 0;
		}
		int fragI,fragJ;
		//System.out.println("step b 3.1");
		int result=isMoveLegal(board,startI,startJ,endI,endJ,board[startI][startJ]);
		//System.out.println("step b 3.2");
		if(result != illegalMove)
		{
			if(Math.abs(endI-startI)==1)
			{
				board[endI][endJ]=board[startI][startJ];
				board[startI][startJ]=MainActivity.empty;
			}
			else if(Math.abs(endI-startI)==2)
			{
				board[(startI+endI)/2][(startJ+endJ)/2]=MainActivity.empty;
				board[endI][endJ]=board[startI][startJ];
				board[startI][startJ]=MainActivity.empty;
			}
			else
			{
				board[endI][endJ]=board[startI][startJ];
				if(startI>endI)
					fragI=-1;
				else
					fragI=1;
				if(startJ>endJ)	
					fragJ=-1;
				else
					fragJ=1;
				board[endI-fragI][endJ-fragJ]=MainActivity.empty;
				board[startI][startJ]=MainActivity.empty;
			}
			if(result==incompleteMove)
			{
				if(!(canCapture(board,endI,endJ)))
					result=legalMove;
			}
			//check for new King
			if(board[endI][endJ]==MainActivity.black && endJ==0)
				board[endI][endJ]=MainActivity.blackKing;
			if(board[endI][endJ]==MainActivity.white && endJ==7)
				board[endI][endJ]=MainActivity.whiteKing;
		}
		return result;
	}	
	
	static int isMoveLegal(int[][] board,int startI,int startJ,int endI,int endJ,int turn)
	{
		int fragi,fragj,tempi,tempj;
		//System.out.println("step b 4.1 startI : "+startI+" startJ : "+startJ+" endI : "+endI+" endJ : "+endJ);
		if(!(inRange(startI,startJ)&& inRange(endI,endJ))){
			//System.out.println("step b 4.2");
			return illegalMove;
		}
		if(board[endI][endJ]!=MainActivity.empty){
			//System.out.println("step b 4.3");
			return illegalMove;
		}
		//System.out.println("step b 4.4");
		int piece=board[startI][startJ];
		if(turn==MainActivity.blackKing || turn==MainActivity.whiteKing)
		{
			//System.out.println("step b 4.5");
			if(canCapture(board,startI,startJ))
			{
				//System.out.println("step b 4.6");
				for(int i=0;i<endList.size();i++)
				{
					int[] intArray=new int[2];
					intArray=(int[])endList.elementAt(i);
					if(intArray[0]==endI && intArray[1]==endJ)
					{
						endList.removeAllElements();
						return incompleteMove;
					}
				}
					return illegalMove;
			}
			else
			{
				//System.out.println("step b 4.7");
				if(canCapture(board,turn))
				{
					//System.out.println("step b 4.8");
					return illegalMove;
				}
				tempi=startI;
				tempj=startJ;
				if(startI>endI)
					fragi=-1;
				else
					fragi=1;
				if(startJ>endJ)	
					fragj=-1;
				else
					fragj=1;
				int check = illegalMove;
				/*do
				{
					System.out.println("step b 4.9 : startI : "+startI+" startJ : "+startJ+" endI : "+endI+" endJ : "+endJ+" tempi : "+tempi+ " tempj : "+tempj);
					tempi=tempi+fragi;
					tempj=tempj+fragj;
					if(color(board[tempi][tempj])==color(board[startI][startJ])){
						System.out.println("step b 4.9.1 : ");
						check = illegalMove;
					}
					else if(tempi==endI && tempj==endJ){
						System.out.println("step b 4.9.2 : ");
						check = legalMove;
					}
				}while(inRange(tempi,tempj));*/
				while(true){
					//System.out.println("step b 4.9.3 c1 : ");
					tempi=tempi+fragi;
					tempj=tempj+fragj;
					if (tempi < 0 || tempi>7 || tempj < 0 || tempj > 7){
						//System.out.println("step b 4.9.3 c2 : ");
						check = illegalMove;
						break;
					}
					else if(color(board[tempi][tempj])==color(board[startI][startJ])){
						//System.out.println("step b 4.9.3 c3 : ");
						check = illegalMove;
						break;
					}
					else if(tempi==endI && tempj==endJ){
						//System.out.println("step b 4.9.3 c4 : ");
						check = legalMove;
						break;
					}
				}
				//System.out.println("step b 4.9.3 c5 : ");
				return check;
				//System.out.println("step b 4.9.3 : ");
				//return check;
			}
			//System.out.println("step b 4.9.3 : ");
			//return illegalMove;
		}
		else if (Math.abs(startI-endI)==1)
		{
			switch(piece)
			{
			case MainActivity.white:
				for(int i=0;i<8;i++)
					for(int j=0;j<8;j++)
					{
						if((board[i][j]==MainActivity.white || board[i][j]==MainActivity.whiteKing) && canCapture(board,i,j))
							return illegalMove;
					}
				/*for(int i=0;i<initList.size();i++)
				{
					int[] intArrays=new int[2];
					intArrays=(int[])initList.elementAt(i);
					if((board[intArrays[0]][intArrays[1]]==MainActivity.white || board[intArrays[0]][intArrays[1]]==MainActivity.whiteKing) && canCapture(board,intArrays[0],intArrays[1]))
						return illegalMove;
				}*/
				if(endJ-startJ==1)
					return legalMove;
				break;
			case MainActivity.black:
				for(int i=0;i<8;i++)
					for(int j=0;j<8;j++)
					{
						if((board[i][j]==MainActivity.black || board[i][j]==MainActivity.blackKing) && canCapture(board,i,j))
							return illegalMove;
					}
				/*for(int i=0;i<initList.size();i++)
				{
					int[] intArrays=new int[2];
					intArrays=(int[])initList.elementAt(i);
					if((board[intArrays[0]][intArrays[1]]==MainActivity.black || board[intArrays[0]][intArrays[1]]==MainActivity.blackKing) && canCapture(board,intArrays[0],intArrays[1]))
						return illegalMove;
				}*/
				if(endJ-startJ==-1)
					return legalMove;
			}
			return illegalMove; 			
		}//else if (Math.abs(startI-endI)==1)
		else if (Math.abs(startI-endI)==2)
		{
			int capI=(startI+endI)/2;
			int capJ=(startJ+endJ)/2;
			int capPiece=board[capI][capJ];
			if(turn==MainActivity.white)
			{
				if(!(capPiece==MainActivity.black || capPiece==MainActivity.blackKing))
					return illegalMove;
			}
			else
			{
				if(!(capPiece==MainActivity.white || capPiece==MainActivity.whiteKing))
					return illegalMove;
			}
			switch(piece)
			{
			case MainActivity.white:
				if(endJ-startJ !=2)
					return illegalMove;
				break;
			case MainActivity.black:
				if(endJ-startJ !=-2)
					return illegalMove;
				break;
			}
			return incompleteMove;
		}
		return illegalMove;
	}
	
	static boolean canCapture(int[][] board,int turn){
		for(int i=0;i<8;i++)
			for(int j=0;j<8;j++)
			{
				if(color(board[i][j])==color(turn) && canCapture(board,i,j))
					return true;
			}//*/
		/*for(int i=0;i<initList.size();i++)
		{
			int[] intArrays=new int[2];
			intArrays=(int[])initList.elementAt(i);
			if(color(board[intArrays[0]][intArrays[1]])==color(turn) && canCapture(board,intArrays[0],intArrays[1]))
				return true;
		}*/
		return false;
	}
	
	static boolean canCapture(int[][] board,int i,int j)
	{
		int fragI=0,fragJ=0,tempI=0,tempJ=0;
		
		switch(board[i][j])
		{
		case MainActivity.white:
			if(i+2<8 && j+2<8)
				if((board[i+1][j+1]==MainActivity.black || board[i+1][j+1]==MainActivity.blackKing) && (board[i+2][j+2]==MainActivity.empty))
					return true;
			if(i-2>-1 && j+2<8)
				if((board[i-1][j+1]==MainActivity.black || board[i-1][j+1]==MainActivity.blackKing) && (board[i-2][j+2]==MainActivity.empty))
					return true;
			break;
		case MainActivity.black:
			if(i+2<8 && j-2>-1)
				if((board[i+1][j-1]==MainActivity.white || board[i+1][j-1]==MainActivity.whiteKing) && (board[i+2][j-2]==MainActivity.empty))
					return true;
			if(i-2>-1 && j-2>-1)
				if((board[i-1][j-1]==MainActivity.white || board[i-1][j-1]==MainActivity.whiteKing) && (board[i-2][j-2]==MainActivity.empty))
					return true;
			break;
		case MainActivity.whiteKing:
		case MainActivity.blackKing:
			endList.removeAllElements();
			for(int k=0;k<4;k++)
			{
				switch(k)
				{
				case 0:
					fragI=1;
					fragJ=1;
					break;
				case 1:
					fragI=-1;
					fragJ=1;
					break;
				case 2:
					fragI=1;
					fragJ=-1;
					break;
				case 3:
					fragI=-1;
					fragJ=-1;
				}
				tempI=i;
				tempJ=j;
				tempI=tempI+fragI;
				tempJ=tempJ+fragJ;
				while(inRange(tempI,tempJ))
				{	
					if(color(board[tempI][tempJ])==opponent(color(board[i][j])))
					{
						if(inRange(tempI+fragI,tempJ+fragJ))
						{
							if(board[tempI+fragI][tempJ+fragJ]==MainActivity.empty)
							{
								int[] intArray=new int[2];
								intArray[0]=tempI+fragI;
								intArray[1]=tempJ+fragJ;
								endList.addElement(intArray);
							}
							else
								break;
						}
						else
							break;
					}
					else if(color(board[i][j])==color(board[tempI][tempJ]))
						break;	
					tempI=tempI+fragI;
					tempJ=tempJ+fragJ;
				}	
			}
			if(endList.size()!=0){
				return true;
			}
			else
				return false;//*/
		}//end switch*/
		return false;
	}
	static int color(int piece)
	{
		switch(piece)
		{
		case MainActivity.white:
		case MainActivity.whiteKing:
			return MainActivity.white;
		case MainActivity.black:
		case MainActivity.blackKing:
			return MainActivity.black;
		}
		return MainActivity.empty;
	}
	private static boolean inRange(int i,int j)
	{
		//System.out.println("step b 4.9.5 : i : "+i+" j : "+j);
		if (i<0 || i>7 || j <0 || j>7){
			return false;
		}
		else{
			return true;
		}
		//return (i>-1 && i<8 && j>-1 && j<8);
	}
	static void moveBoard(int[][] board,int[] move,boolean realmove)
	{
		if (realmove){
			if (board[move[0]][move[1]] == MainActivity.blackKing || board[move[0]][move[1]] == MainActivity.whiteKing)
				MainActivity.countDraw++;
			else
				MainActivity.countDraw = 0;
		}
		int startx=move[0];
		int starty=move[1];
		int endx=move[2];
		int endy=move[3];
		while(endx>0 || endy>0)
		{
			applyMove(board,startx,starty,endx%10,endy%10,false);
			startx=endx%10;
			starty=endy%10;
			endx /=10;
			endy /=10; 
		}
		
	}
	
	static Vector<int[]> generateMove(int[][] board,int turn)
	{
		Vector<int[]> movesList=new Vector<int[]>();
		int fragI=0,fragJ=0,tempI,tempJ;
		int move;
		boolean checkCanCapture1=false;
		boolean checkCanCapture2=false;
		//c
		//System.out.println("step size :"+initList.size());
		/*for(int i=0;i<initList.size();i++)
		{
			int[] intArrays=new int[2];
			intArrays=(int[])initList.elementAt(i);
			if(color(turn)==color(board[intArrays[0]][intArrays[1]]))
			{
				if(board[intArrays[0]][intArrays[1]]<=2)
				{
					 if(canCapture(board,intArrays[0],intArrays[1]))
					 {
						 if (checkCanCapture1==false)
					     {
						    checkCanCapture1=true;
							if(checkCanCapture2==false) 
						       movesList.removeAllElements();
					     }
						 for(int k=-2;k<=2;k+=4) 
							for(int l=-2;l<=2;l+=4)
							{
								move=isMoveLegal(board,intArrays[0],intArrays[1],intArrays[0]+k,intArrays[1]+l,turn);
								if(move==incompleteMove)
								{
									int[] intArray=new int[4];
									intArray[0]=intArrays[0];
									intArray[1]=intArrays[1];
									intArray[2]=intArrays[0]+k;
									intArray[3]=intArrays[1]+l;
									int[][] tempBoard=copyBoard(board);
									move=applyMove(tempBoard,intArrays[0],intArrays[1],intArrays[0]+k,intArrays[1]+l,false);
									if(move==incompleteMove) 
										forceCapture1(tempBoard,turn,intArray,movesList,10);
									else{
										movesList.addElement(intArray);
									}
								}
							}
					}//end  if(canCapture(board,turn))
					else if(checkCanCapture1==false && checkCanCapture2==false)
					{
						for(int k=-1;k<=2;k+=2)
							for(int l=-1;l<=2;l+=2)
							{
								if(inRange(intArrays[0]+k,intArrays[1]+l))
								{
									move=isWalkLegal(board,intArrays[0],intArrays[1],intArrays[0]+k,intArrays[1]+l,turn);
									if(move==legalMove)
									{
										int[] intArray = new int[4];
										intArray[0]=intArrays[0];
										intArray[1]=intArrays[1];
										intArray[2]=intArrays[0]+k;
										intArray[3]=intArrays[1]+l;
										movesList.addElement(intArray);
									}
								}
							}
					}
				}//end if(color(turn)==board[i][j])
				else
				{
					if(canCapture(board,intArrays[0],intArrays[1]))//if(canCapture(board,turn))////////edit 1
					{
						if (checkCanCapture2==false)
					    {
						    checkCanCapture2=true;
							if(checkCanCapture1==false)
						       movesList.removeAllElements();
					    }
						for(int m=0;m<endList.size();m++)
						{
						  int[] spArray=new int[2];
				          spArray=(int[])endList.elementAt(m);
				          int spI=spArray[0];
						  int spJ=spArray[1];
						  int[] intArray = new int[4];
						  intArray[0]=intArrays[0];
						  intArray[1]=intArrays[1];
						  intArray[2]=spI;
						  intArray[3]=spJ;
						  int[][] tempBoard=copyBoard(board); 
					      move=applyMove(tempBoard,intArrays[0],intArrays[1],spI,spJ,false);
					     
						  if(move==incompleteMove) // มีการเดินไปกิน
							 forceCapture2(tempBoard,turn,intArray,movesList,10);
						  else
							 movesList.addElement(intArray);
						}
					}
					else if(checkCanCapture1==false && checkCanCapture2==false)
					{
						for(int k=0;k<4;k++)
						{
							switch(k)
							{
							case 0:
								fragI=1;
								fragJ=1;
								break;
							case 1:
								fragI=-1;
								fragJ=1;
								break;
							case 2:
								fragI=1;
								fragJ=-1;
								break;
							case 3:
								fragI=-1;
								fragJ=-1;
							}//end switch

							tempI=intArrays[0];
							tempJ=intArrays[1];
							tempI=tempI+fragI;
							tempJ=tempJ+fragJ;
							while(inRange(tempI,tempJ))
							{	
								if(board[tempI][tempJ]==MainActivity.empty)
								{
									int[] intArray=new int[4];
									intArray[0]=intArrays[0];
									intArray[1]=intArrays[1];
									intArray[2]=tempI;
									intArray[3]=tempJ;		
									movesList.addElement(intArray);
								}
								else
									break;	
								tempI=tempI+fragI;
								tempJ=tempJ+fragJ;
							}
						}//end for
					}//end else cap
				}//end else
			}
		}*/
		
		for(int i=7;i>=0;i--)
			for(int j=0;j<8;j++)
			{
				if(color(turn)==color(board[i][j]))
				{
					if(board[i][j]<=2)
					{
						 if(canCapture(board,i,j))
						 {
							 if (checkCanCapture1==false)
						     {
							    checkCanCapture1=true;
								if(checkCanCapture2==false) 
							       movesList.removeAllElements();
						     }
							 for(int k=-2;k<=2;k+=4) 
								for(int l=-2;l<=2;l+=4)
								{
									move=isMoveLegal(board,i,j,i+k,j+l,turn);
									if(move==incompleteMove)
									{
										int[] intArray=new int[4];
										intArray[0]=i;
										intArray[1]=j;
										intArray[2]=i+k;
										intArray[3]=j+l;
										int[][] tempBoard=copyBoard(board);
										move=applyMove(tempBoard,i,j,i+k,j+l,false);
										if(move==incompleteMove) 
											forceCapture1(tempBoard,turn,intArray,movesList,10);
										else{
											movesList.addElement(intArray);
										}
									}
								}
						}//end  if(canCapture(board,turn))
						else if(checkCanCapture1==false && checkCanCapture2==false)
						{
							for(int k=-1;k<=2;k+=2)
								for(int l=-1;l<=2;l+=2)
								{
									if(inRange(i+k,j+l))
									{
										move=isWalkLegal(board,i,j,i+k,j+l,turn);
										if(move==legalMove)
										{
											int[] intArray = new int[4];
											intArray[0]=i;
											intArray[1]=j;
											intArray[2]=i+k;
											intArray[3]=j+l;
											movesList.addElement(intArray);
										}
									}
								}
						}
					}//end if(color(turn)==board[i][j])
					else
					{
						if(canCapture(board,i,j))//if(canCapture(board,turn))////////edit 1
						{
							if (checkCanCapture2==false)
						    {
							    checkCanCapture2=true;
								if(checkCanCapture1==false)
							       movesList.removeAllElements();
						    }
							for(int m=0;m<endList.size();m++)
							{
							  int[] spArray=new int[2];
					          spArray=(int[])endList.elementAt(m);
					          int spI=spArray[0];
							  int spJ=spArray[1];
							  int[] intArray = new int[4];
							  intArray[0]=i;
							  intArray[1]=j;
							  intArray[2]=spI;
							  intArray[3]=spJ;
							  int[][] tempBoard=copyBoard(board); 
						      move=applyMove(tempBoard,i,j,spI,spJ,false);
						     
							  if(move==incompleteMove) // มีการเดินไปกิน
								 forceCapture2(tempBoard,turn,intArray,movesList,10);
							  else
								 movesList.addElement(intArray);
							}
						}
						else if(checkCanCapture1==false && checkCanCapture2==false)
						{
							for(int k=0;k<4;k++)
							{
								switch(k)
								{
								case 0:
									fragI=1;
									fragJ=1;
									break;
								case 1:
									fragI=-1;
									fragJ=1;
									break;
								case 2:
									fragI=1;
									fragJ=-1;
									break;
								case 3:
									fragI=-1;
									fragJ=-1;
								}//end switch
 
								tempI=i;
								tempJ=j;
								tempI=tempI+fragI;
								tempJ=tempJ+fragJ;
								while(inRange(tempI,tempJ))
								{	
									if(board[tempI][tempJ]==MainActivity.empty)
									{
										int[] intArray=new int[4];
										intArray[0]=i;
										intArray[1]=j;
										intArray[2]=tempI;
										intArray[3]=tempJ;		
										movesList.addElement(intArray);
									}
									else
										break;	
									tempI=tempI+fragI;
									tempJ=tempJ+fragJ;
								}
							}//end for
						}//end else cap
					}//end else
				}
			}	//end for
		//	*/
		checkCanCapture1=false;
		checkCanCapture2=false;
		return movesList;
	}
	
	private static void forceCapture2(int[][] board,int turn,int[] move,Vector<int[]> movesList,int inc)
	{
		int newx=move[2];
		int newy=move[3];

		while(newx>7 || newy>7)
		{
			newx/=10;
			newy/=10;
		}
		if(canCapture(board,newx,newy))
		{
			for(int i=0;i<endList.size();i++)
			{	
				int[] spArray=new int[2];
				spArray=(int[])endList.elementAt(i);
				int spI=spArray[0];
				int spJ=spArray[1];
				int[][] tempBoard=copyBoard(board);/////******* forget set sp again
				int moveResult=applyMove(tempBoard,newx,newy,spI,spJ,false);/// after if canCap we have endList
				if(moveResult==legalMove)
				{
					int[] newMove=new int[4];
					newMove[0]=move[0];
					newMove[1]=move[1];
					newMove[2]=move[2]+(spI)*inc;
					newMove[3]=move[3]+(spJ)*inc;
					movesList.addElement(newMove);
				}
				else if(moveResult==incompleteMove)
				{
					int[] newMove=new int[4];
					newMove[0]=move[0];
					newMove[1]=move[1];
					newMove[2]=move[2]+(spI)*inc;
					newMove[3]=move[3]+(spJ)*inc;
					forceCapture2(tempBoard,turn,newMove,movesList,inc*10);
				}
			}// end for endList
		}
	}
	private static void forceCapture1(int[][] board,int turn,int[] move,Vector<int[]> movesList,int inc)
	{
		int newx=move[2];
		int newy=move[3];
		//int opponent;
		while(newx>7 || newy>7)
		{
			newx/=10;
			newy/=10;
		}
		for(int i=-2;i<=2;i+=4)
			for(int j=-2;j<=2;j+=4)
			{
				if(inRange(newx+i,newy+j))
				{
					int[][] tempBoard=copyBoard(board);
					int moveResult=applyMove(tempBoard,newx,newy,newx+i,newy+j,false);
					if(moveResult==legalMove)
					{
						int[] newMove=new int[4];
						newMove[0]=move[0];
						newMove[1]=move[1];
						newMove[2]=move[2]+(newx+i)*inc;
						newMove[3]=move[3]+(newy+j)*inc;
						movesList.addElement(newMove);
					}
					else if(moveResult==incompleteMove)
					{
						int[] newMove=new int[4];
						newMove[0]=move[0];
						newMove[1]=move[1];
						newMove[2]=move[2]+(newx+i)*inc;
						newMove[3]=move[3]+(newy+j)*inc;
						forceCapture1(tempBoard,turn,newMove,movesList,inc*10);
					}
				}
			}
	}
	
	static int isWalkLegal(int[][] board,int startI,int startJ,int endI,int endJ,int turn)
	{
		if(!(inRange(startI,startJ)&& inRange(endI,endJ)))
			return illegalMove;
		if(board[endI][endJ]!=MainActivity.empty)
			return illegalMove;
		int piece=board[startI][startJ];
		if(Math.abs(startI-endI)==1)
		{
			switch(piece)
			{
			case MainActivity.white:
				if(endJ-startJ==1)
					return legalMove;
					break;
			case MainActivity.black:
				if(endJ-startJ==-1)
					return legalMove;
				break;
			}
			return illegalMove;
		}
		return illegalMove;
	}
	
	static boolean noMoves(int[][] board,int toMove)
	{
		for(int i=0;i<8;i++)
			for(int j=0;j<8;j++)
				if((float)(i+j)/2 != (i+j)/2)
				{
					if(toMove== MainActivity.white &&(color(board[i][j])==MainActivity.white))
					{
						if(canWalk(board,i,j)) return false;
						else if(canCapture(board,i,j)) return false;
					}
					else if(toMove==MainActivity.black &&(color(board[i][j])==MainActivity.black))
					{
						if(canWalk(board,i,j)) return false;
						else if(canCapture(board,i,j)) return false;
					}
				}//*/
		/*for(int i=0;i<initList.size();i++)
		{
			int[] intArrays=new int[2];
			intArrays=(int[])initList.elementAt(i);
			if(toMove== MainActivity.white &&(color(board[intArrays[0]][intArrays[1]])==MainActivity.white))
			{
				if(canWalk(board,intArrays[0],intArrays[1])) return false;
				else if(canCapture(board,intArrays[0],intArrays[1])) return false;
			}
			else if(toMove==MainActivity.black &&(color(board[intArrays[0]][intArrays[1]])==MainActivity.black))
			{
				if(canWalk(board,intArrays[0],intArrays[1])) return false;
				else if(canCapture(board,intArrays[0],intArrays[1])) return false;
			}
		}*/
		return true;
	}
	static boolean canWalk(int[][] board,int i,int j)
	{
		int fragI=0,fragJ=0,tempI,tempJ;
		switch(board[i][j])
		{
		case MainActivity.white :
			if(isEmpty(board,i+1,j+1) || isEmpty(board,i-1,j+1))
				return true;
			break;
		case MainActivity.black:
			if(isEmpty(board,i+1,j-1) || isEmpty(board,i-1,j-1))
				return true;
			break;
		case MainActivity.whiteKing :
		case MainActivity.blackKing :
			for(int k=0;k<4;k++)
			{
				switch(k)
				{
					case 0:
						fragI=1;
						fragJ=1;
						break;
					case 1:
						fragI=-1;
						fragJ=1;
						break;
					case 2:
						fragI=1;
						fragJ=-1;
						break;
					case 3:
						fragI=-1;
						fragJ=-1;
				}//end switch
				tempI=i;
				tempJ=j;
				tempI=tempI+fragI;
				tempJ=tempJ+fragJ;
				while(inRange(tempI,tempJ))
				{	
					if(board[tempI][tempJ]==MainActivity.empty)
						return true;
					tempI=tempI+fragI;
					tempJ=tempJ+fragJ;
				}
			}//end for	
		}
		return false;
	}
	private static boolean isEmpty(int[][] board,int i,int j)
	{
		if(i>-1 && i<8 && j>-1 && j<8)
			if(board[i][j] == MainActivity.empty)
				return true;
		return false;
	}
	static boolean checkDraw(int[][] board){
		int w=0,b=0;
		for(int i=0;i<8;i++){
			for(int j=0;j<8;j++){
				if(color(board[i][j])==MainActivity.white){
				 	w=w+1;
					if(w>2){
						w=0;
						return false;
					}
				}
				if(color(board[i][j])==MainActivity.black){
					b=b+1;
					if(b>2){
						b=0;
						return false;
					}
				}
			}
		}
		/*for(int i=0;i<initList.size();i++)
		{
			int[] intArrays=new int[2];
			intArrays=(int[])initList.elementAt(i);
			if(color(board[intArrays[0]][intArrays[1]])==MainActivity.white){
			 	w=w+1;
				if(w>2){
					w=0;
					return false;
				}
			}
			if(color(board[intArrays[0]][intArrays[1]])==MainActivity.black){
				b=b+1;
				if(b>2){
					b=0;
					return false;
				}
			}
		}*/
		if(w==1 && b==1){
			ch1=ch1+1;
			if(ch1==18){
				ch1=0;	
				return true;
			}else
				return false;
		}else if(w==2 && b==2){
			ch2=ch2+1;
			if(ch2==50){
				ch2=0;	
				return true;
			}else
				return false;		
		}
		return false;
	}
	static int opponent(int turn)
	{
		switch(turn)
		{
		case MainActivity.black:
		case MainActivity.blackKing:
			return MainActivity.white;
		case MainActivity.white:
		case MainActivity.whiteKing:
			return MainActivity.black;
		}
		return MainActivity.empty;
	}
	static int[][] copyBoard(int[][] board)
	{
		int[][] copy=new int[8][8];
		for(int i=0;i<8;i++)
			System.arraycopy(board[i],0,copy[i],0,8);
		return copy;
	}
}
